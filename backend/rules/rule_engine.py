"""
Rule-Based Fraud Engine
Hard rules that override or augment ML scores.
"""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RuleResult:
    def __init__(self, rule_id: str, triggered: bool, score_delta: float, reason: str):
        self.rule_id = rule_id
        self.triggered = triggered
        self.score_delta = score_delta
        self.reason = reason


class FraudRuleEngine:
    """
    Evaluates hard business rules against transaction features.
    Returns a rule score (0-10) and triggered reasons.
    """

    RULES = [
        "R001_VELOCITY_BURST",
        "R002_HIGH_RISK_COUNTRY",
        "R003_NEW_DEVICE_HIGH_AMOUNT",
        "R004_VPN_PROXY_DETECTED",
        "R005_UNUSUAL_HOUR_LARGE_TX",
        "R006_ROUND_AMOUNT_HIGH_FREQ",
        "R007_GEO_MISMATCH",
        "R008_HIGH_RISK_MERCHANT",
        "R009_NEW_ACCOUNT_HIGH_VALUE",
        "R010_BEHAVIORAL_DRIFT_SEVERE",
        "R011_SESSION_RAPID_FIRE",
        "R012_KNOWN_BAD_IP",
    ]

    def evaluate(self, features: Dict[str, float], transaction: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Returns (rule_score 0-10, list of triggered rule reasons).
        """
        results = []
        
        results.append(self._r001_velocity_burst(features))
        results.append(self._r002_high_risk_country(features, transaction))
        results.append(self._r003_new_device_high_amount(features, transaction))
        results.append(self._r004_vpn_proxy(features))
        results.append(self._r005_unusual_hour_large(features, transaction))
        results.append(self._r006_round_amount_freq(features))
        results.append(self._r007_geo_mismatch(features))
        results.append(self._r008_high_risk_merchant(features, transaction))
        results.append(self._r009_new_account_high_value(features, transaction))
        results.append(self._r010_behavioral_drift(features))
        results.append(self._r011_rapid_fire_session(features))

        triggered = [r for r in results if r.triggered]
        reasons = [r.reason for r in triggered]
        
        total_delta = sum(r.score_delta for r in triggered)
        rule_score = min(total_delta, 10.0)

        return rule_score, reasons

    def _r001_velocity_burst(self, f: Dict) -> RuleResult:
        triggered = f.get("tx_count_1h", 0) > 8
        return RuleResult("R001", triggered, 3.5 if triggered else 0, 
                         f"High transaction velocity: {f.get('tx_count_1h', 0):.0f} transactions in 1h")

    def _r002_high_risk_country(self, f: Dict, t: Dict) -> RuleResult:
        triggered = f.get("is_high_risk_country", 0) == 1.0
        country = t.get("country", "?")
        return RuleResult("R002", triggered, 2.0 if triggered else 0,
                         f"Transaction from high-risk country: {country}")

    def _r003_new_device_high_amount(self, f: Dict, t: Dict) -> RuleResult:
        is_new = f.get("is_new_device", 0) == 1.0
        amount = t.get("amount", 0)
        triggered = is_new and amount > 1000
        return RuleResult("R003", triggered, 3.0 if triggered else 0,
                         f"New device with high transaction amount: ${amount:.2f}")

    def _r004_vpn_proxy(self, f: Dict) -> RuleResult:
        triggered = f.get("is_vpn_proxy", 0) == 1.0
        return RuleResult("R004", triggered, 1.5 if triggered else 0,
                         "VPN/Proxy IP detected")

    def _r005_unusual_hour_large(self, f: Dict, t: Dict) -> RuleResult:
        unusual = f.get("unusual_hour", 0) == 1.0
        amount = t.get("amount", 0)
        triggered = unusual and amount > 500
        return RuleResult("R005", triggered, 2.0 if triggered else 0,
                         f"High-value transaction during unusual hours: ${amount:.2f}")

    def _r006_round_amount_freq(self, f: Dict) -> RuleResult:
        is_round = f.get("is_round_amount", 0) == 1.0
        high_freq = f.get("tx_count_1h", 0) > 3
        triggered = is_round and high_freq
        return RuleResult("R006", triggered, 1.5 if triggered else 0,
                         "Round-amount structuring pattern detected")

    def _r007_geo_mismatch(self, f: Dict) -> RuleResult:
        triggered = f.get("geo_mismatch", 0) > 0.5 and f.get("distance_from_home", 0) > 0.5
        return RuleResult("R007", triggered, 2.5 if triggered else 0,
                         "Geographic location mismatch from user's home region")

    def _r008_high_risk_merchant(self, f: Dict, t: Dict) -> RuleResult:
        triggered = f.get("merchant_risk", 0) == 1.0
        cat = t.get("merchant_category", "")
        return RuleResult("R008", triggered, 2.0 if triggered else 0,
                         f"High-risk merchant category: {cat}")

    def _r009_new_account_high_value(self, f: Dict, t: Dict) -> RuleResult:
        age = f.get("account_age_days", 365)
        amount = t.get("amount", 0)
        triggered = age < 30 and amount > 2000
        return RuleResult("R009", triggered, 3.5 if triggered else 0,
                         f"New account (<30 days) with high-value transaction: ${amount:.2f}")

    def _r010_behavioral_drift(self, f: Dict) -> RuleResult:
        triggered = f.get("behavioral_drift", 0) > 0.7
        return RuleResult("R010", triggered, 2.0 if triggered else 0,
                         f"Severe behavioral drift detected (score: {f.get('behavioral_drift', 0):.2f})")

    def _r011_rapid_fire_session(self, f: Dict) -> RuleResult:
        triggered = f.get("session_anomaly", 0) == 1.0
        return RuleResult("R011", triggered, 2.5 if triggered else 0,
                         "Rapid-fire transactions in same session (<30s apart)")
