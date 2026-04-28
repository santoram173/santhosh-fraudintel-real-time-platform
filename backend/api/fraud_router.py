"""
Fraud Scoring API Router
POST /api/v1/fraud/score  - Score a single transaction
POST /api/v1/fraud/batch  - Score multiple transactions
GET  /api/v1/fraud/audit  - Get recent audit log
"""

import time
import uuid
import sys
import os
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException, Depends
from typing import List

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from backend.models.schemas import (
    TransactionRequest, FraudScoreResponse, BatchTransactionRequest
)
from backend.core.fraud_scorer import RiskFusionEngine
from backend.rules.rule_engine import FraudRuleEngine
from backend.explainability.explainer import generate_explanation
from ml.features.feature_engineering import extract_features, features_to_vector
from backend.core.identity_graph import get_graph

router = APIRouter()
_rule_engine = FraudRuleEngine()
_risk_engine = RiskFusionEngine()


async def get_model(request: Request):
    registry = request.app.state.model_registry
    return registry.get_fraud_model()


async def get_audit(request: Request):
    return request.app.state.audit_logger


@router.post("/score", response_model=FraudScoreResponse)
async def score_transaction(
    tx: TransactionRequest,
    request: Request,
):
    """
    Score a single transaction in real-time.
    Target: sub-100ms response time.
    """
    start = time.perf_counter()
    
    transaction_id = tx.transaction_id or str(uuid.uuid4())
    tx_dict = tx.dict()
    tx_dict["transaction_id"] = transaction_id

    # 1. Feature extraction
    features = extract_features(tx_dict)
    feature_vector = features_to_vector(features)

    # 2. ML scoring
    model = await get_model(request)
    ml_score = model.predict_score(feature_vector)

    # 3. Rule engine
    rule_score, rule_reasons = _rule_engine.evaluate(features, tx_dict)

    # 4. Identity graph
    graph = get_graph()
    identity_result = graph.record_event(tx.user_id, tx.device_id, tx.ip_address)
    identity_risk = identity_result["composite_risk"]

    # 5. Risk fusion
    risk_score = _risk_engine.fuse(ml_score, rule_score, identity_risk)
    decision = _risk_engine.decide(risk_score)

    # 6. Explanation
    explanation = generate_explanation(
        features, rule_reasons, ml_score, 
        identity_result.get("anomaly_signals", []), risk_score
    )

    # 7. Audit log
    audit = await get_audit(request)
    audit.log_decision(
        transaction_id=transaction_id,
        decision=decision,
        risk_score=risk_score,
        ml_score=ml_score,
        rule_score=rule_score,
        identity_risk=identity_risk,
        explanation=explanation,
        user_id=tx.user_id,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return FraudScoreResponse(
        transaction_id=transaction_id,
        decision=decision,
        risk_score=round(risk_score, 2),
        ml_score=round(ml_score, 4),
        rule_score=round(rule_score, 2),
        identity_risk=round(identity_risk, 4),
        explanation=explanation,
        features_used={k: round(v, 3) for k, v in list(features.items())[:12]},
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/batch", response_model=List[FraudScoreResponse])
async def score_batch(batch: BatchTransactionRequest, request: Request):
    """Score multiple transactions (up to 100)."""
    if len(batch.transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 transactions")
    
    results = []
    for tx in batch.transactions:
        result = await score_transaction(tx, request)
        results.append(result)
    return results


@router.get("/audit")
async def get_audit_log(limit: int = 50, request: Request = None):
    """Retrieve recent audit log entries."""
    audit = await get_audit(request)
    return {"entries": audit.get_recent(limit), "total": limit}


@router.get("/audit/{transaction_id}")
async def get_transaction_audit(transaction_id: str, request: Request):
    """Get audit trail for a specific transaction."""
    audit = await get_audit(request)
    entries = audit.get_by_transaction(transaction_id)
    if not entries:
        raise HTTPException(status_code=404, detail="Transaction not found in audit log")
    return {"transaction_id": transaction_id, "entries": entries}
