"""Audit Logger - immutable audit trail for all fraud decisions."""

import json
import time
import uuid
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

AUDIT_FILE = Path("data/audit_log.jsonl")


class AuditLogger:
    def __init__(self):
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._log: List[Dict] = []

    def log_decision(self, transaction_id: str, decision: str, risk_score: float,
                     ml_score: float, rule_score: float, identity_risk: float,
                     explanation: List[str], user_id: str, analyst_id: str = None):
        entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "transaction_id": transaction_id,
            "user_id": user_id,
            "decision": decision,
            "risk_score": round(risk_score, 3),
            "ml_score": round(ml_score, 4),
            "rule_score": round(rule_score, 3),
            "identity_risk": round(identity_risk, 4),
            "explanation": explanation,
            "analyst_id": analyst_id,
            "source": "automated" if not analyst_id else "analyst",
        }
        self._log.append(entry)
        self._write(entry)
        return entry

    def log_case_update(self, case_id: str, analyst_id: str, action: str, notes: str):
        entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "type": "case_update",
            "case_id": case_id,
            "analyst_id": analyst_id,
            "action": action,
            "notes": notes,
        }
        self._log.append(entry)
        self._write(entry)
        return entry

    def _write(self, entry: Dict):
        try:
            with open(AUDIT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Audit write failed: {e}")

    def get_recent(self, limit: int = 100) -> List[Dict]:
        return self._log[-limit:]

    def get_by_transaction(self, transaction_id: str) -> List[Dict]:
        return [e for e in self._log if e.get("transaction_id") == transaction_id]
