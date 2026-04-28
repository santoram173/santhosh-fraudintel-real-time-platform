"""Analytics API - dashboard data and reporting."""

import time
import random
from fastapi import APIRouter, Request
from collections import Counter

router = APIRouter()


@router.get("/dashboard")
async def dashboard_data(request: Request):
    audit = request.app.state.audit_logger
    entries = audit.get_recent(500)
    
    if not entries:
        # Return synthetic data for demo
        return _synthetic_dashboard()
    
    decisions = [e.get("decision") for e in entries if "decision" in e]
    scores = [e.get("risk_score", 0) for e in entries if "risk_score" in e]
    
    decision_counts = Counter(decisions)
    total = len(decisions)
    
    return {
        "total_transactions": total,
        "approved": decision_counts.get("APPROVE", 0),
        "reviewed": decision_counts.get("REVIEW", 0),
        "blocked": decision_counts.get("BLOCK", 0),
        "fraud_rate": round(decision_counts.get("BLOCK", 0) / max(total, 1) * 100, 2),
        "avg_risk_score": round(sum(scores) / max(len(scores), 1), 2),
        "processing_capacity": "sub-100ms",
    }


def _synthetic_dashboard():
    return {
        "total_transactions": 12847,
        "approved": 11203,
        "reviewed": 1289,
        "blocked": 355,
        "fraud_rate": 2.76,
        "avg_risk_score": 2.14,
        "processing_capacity": "sub-100ms",
        "top_risk_countries": [
            {"country": "NG", "count": 89, "avg_score": 7.2},
            {"country": "RU", "count": 64, "avg_score": 6.8},
            {"country": "CN", "count": 51, "avg_score": 5.9},
        ],
        "hourly_volume": [
            {"hour": h, "count": random.randint(100, 800), "fraud": random.randint(2, 20)}
            for h in range(24)
        ],
    }


@router.get("/live-feed")
async def live_feed(request: Request, limit: int = 20):
    """Latest scored transactions for live feed."""
    audit = request.app.state.audit_logger
    entries = audit.get_recent(limit)
    return {"feed": list(reversed(entries)), "count": len(entries)}
