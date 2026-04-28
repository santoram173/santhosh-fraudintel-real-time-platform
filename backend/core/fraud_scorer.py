"""
Risk Fusion Scoring Engine
Fuses ML score + Rule score + Identity risk into final 0-10 risk score.
"""

import time
import uuid
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

WEIGHTS = {
    "ml_score": 0.45,
    "rule_score": 0.35,
    "identity_risk": 0.20,
}

THRESHOLDS = {
    "APPROVE": 3.5,
    "REVIEW": 6.5,
    "BLOCK": 10.1,
}


class RiskFusionEngine:
    """Fuses all signal sources into a final risk decision."""

    def fuse(
        self,
        ml_score: float,       # 0-1
        rule_score: float,     # 0-10
        identity_risk: float,  # 0-1
    ) -> float:
        """
        Weighted fusion → final risk score 0-10.
        """
        # Normalize rule_score to 0-1
        rule_normalized = rule_score / 10.0

        # Weighted average
        fused = (
            WEIGHTS["ml_score"] * ml_score
            + WEIGHTS["rule_score"] * rule_normalized
            + WEIGHTS["identity_risk"] * identity_risk
        )

        # Scale to 0-10
        risk_score = fused * 10.0

        # Hard boosts for critical signals
        if ml_score > 0.85 or rule_score >= 7.0:
            risk_score = max(risk_score, 7.5)
        if ml_score > 0.95 or rule_score >= 9.0:
            risk_score = max(risk_score, 9.0)

        return float(min(risk_score, 10.0))

    def decide(self, risk_score: float) -> str:
        if risk_score < THRESHOLDS["APPROVE"]:
            return "APPROVE"
        elif risk_score < THRESHOLDS["REVIEW"]:
            return "REVIEW"
        else:
            return "BLOCK"
