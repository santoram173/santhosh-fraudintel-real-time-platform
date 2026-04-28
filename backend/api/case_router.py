"""Case Management Router - analyst review workflow."""

import time
import uuid
from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from backend.models.schemas import Case, CaseCreateRequest, CaseUpdateRequest, CaseStatus, Decision

router = APIRouter()

# In-memory case store (replace with DB in production)
_cases: dict = {}


@router.post("/", response_model=Case)
async def create_case(req: CaseCreateRequest, request: Request):
    case_id = f"CASE-{str(uuid.uuid4())[:8].upper()}"
    now = time.time()
    case = Case(
        case_id=case_id,
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        risk_score=req.risk_score,
        decision=req.decision,
        status=CaseStatus.OPEN,
        created_at=now,
        updated_at=now,
        notes=req.notes,
        audit_trail=[{
            "action": "CASE_CREATED",
            "timestamp": now,
            "decision": req.decision,
            "risk_score": req.risk_score,
        }]
    )
    _cases[case_id] = case
    
    audit = request.app.state.audit_logger
    audit.log_case_update(case_id, "system", "CREATE", f"Case created for tx {req.transaction_id}")
    
    return case


@router.get("/", response_model=List[Case])
async def list_cases(
    status: Optional[CaseStatus] = None,
    decision: Optional[Decision] = None,
    limit: int = 50,
):
    cases = list(_cases.values())
    if status:
        cases = [c for c in cases if c.status == status]
    if decision:
        cases = [c for c in cases if c.decision == decision]
    cases.sort(key=lambda c: c.created_at, reverse=True)
    return cases[:limit]


@router.get("/{case_id}", response_model=Case)
async def get_case(case_id: str):
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found")
    return _cases[case_id]


@router.patch("/{case_id}", response_model=Case)
async def update_case(case_id: str, update: CaseUpdateRequest, request: Request):
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case = _cases[case_id]
    now = time.time()
    
    audit_entry = {
        "action": f"STATUS_CHANGE_{update.status}",
        "timestamp": now,
        "analyst_id": update.analyst_id,
        "previous_status": case.status,
        "new_status": update.status,
        "notes": update.notes,
    }
    
    case.status = update.status
    case.analyst_id = update.analyst_id
    case.notes = update.notes
    case.updated_at = now
    
    if update.decision_override:
        audit_entry["decision_override"] = update.decision_override
        case.decision = update.decision_override
    
    case.audit_trail.append(audit_entry)
    
    audit = request.app.state.audit_logger
    audit.log_case_update(case_id, update.analyst_id, f"UPDATE_{update.status}", update.notes)
    
    return case


@router.get("/stats/summary")
async def case_stats():
    cases = list(_cases.values())
    return {
        "total": len(cases),
        "open": sum(1 for c in cases if c.status == CaseStatus.OPEN),
        "in_review": sum(1 for c in cases if c.status == CaseStatus.IN_REVIEW),
        "closed": sum(1 for c in cases if c.status == CaseStatus.CLOSED),
        "escalated": sum(1 for c in cases if c.status == CaseStatus.ESCALATED),
        "avg_risk_score": round(sum(c.risk_score for c in cases) / max(len(cases), 1), 2),
        "high_risk": sum(1 for c in cases if c.risk_score >= 7.0),
    }
