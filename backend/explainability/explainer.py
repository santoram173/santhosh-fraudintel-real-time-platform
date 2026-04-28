"""
Explainability Module
Generates human-readable explanations for fraud decisions.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

FEATURE_DESCRIPTIONS = {
    "tx_count_1h": ("High transaction count in last hour", 0.7),
    "velocity_spike": ("Velocity spike detected", 0.6),
    "is_new_device": ("Transaction from new/unrecognized device", 0.5),
    "is_high_risk_country": ("Transaction from high-risk country", 0.5),
    "geo_mismatch": ("Geographic location mismatch", 0.5),
    "distance_from_home": ("Transaction far from user's home location", 0.4),
    "ip_risk": ("Elevated IP reputation risk", 0.4),
    "is_vpn_proxy": ("VPN or proxy IP detected", 0.4),
    "behavioral_drift": ("Behavioral pattern change detected", 0.4),
    "unusual_hour": ("Transaction at unusual hour", 0.3),
    "amount_vs_avg": ("Transaction amount deviates significantly from user average", 0.4),
    "is_large_amount": ("Large transaction amount", 0.3),
    "merchant_risk": ("High-risk merchant category", 0.4),
    "session_anomaly": ("Suspicious session activity", 0.5),
    "account_age_normalized": ("Account is relatively new", 0.3, True),  # inverted
    "device_risk_score": ("Device has elevated risk score", 0.4),
}


def generate_explanation(
    features: Dict[str, float],
    rule_reasons: List[str],
    ml_score: float,
    identity_signals: List[str],
    risk_score: float,
) -> List[str]:
    """
    Generates ordered list of human-readable explanation strings.
    """
    explanations = []

    # Add rule-based reasons first (highest priority)
    for reason in rule_reasons[:5]:
        explanations.append(f"[RULE] {reason}")

    # Add ML feature explanations
    ml_reasons = _ml_feature_reasons(features, ml_score)
    for reason in ml_reasons[:4]:
        explanations.append(f"[ML] {reason}")

    # Add identity graph signals
    for signal in identity_signals[:3]:
        explanations.append(f"[IDENTITY] {signal}")

    # Summary if no specific reasons
    if not explanations:
        if risk_score < 3.5:
            explanations.append("Transaction matches normal user behavior patterns")
        elif risk_score < 6.5:
            explanations.append("Some risk signals present; manual review recommended")
        else:
            explanations.append("Multiple high-severity fraud indicators detected")

    return explanations[:8]  # Cap at 8 reasons


def _ml_feature_reasons(features: Dict[str, float], ml_score: float) -> List[str]:
    reasons = []
    
    # Check each high-signal feature
    if features.get("tx_count_1h", 0) > 5:
        reasons.append(f"High transaction velocity: {features['tx_count_1h']:.0f} transactions in last hour")
    
    if features.get("velocity_spike", 0) > 0.6:
        reasons.append("Transaction velocity significantly above user baseline")
    
    if features.get("is_new_device", 0) == 1.0:
        reasons.append("Transaction initiated from unrecognized device")
    
    if features.get("geo_mismatch", 0) > 0.5:
        reasons.append("Transaction location inconsistent with user's typical geography")
    
    if features.get("behavioral_drift", 0) > 0.5:
        reasons.append(f"Behavioral anomaly score: {features['behavioral_drift']:.2f} (threshold: 0.5)")
    
    if features.get("ip_risk", 0) > 0.5:
        reasons.append("IP address associated with elevated fraud activity")
    
    if features.get("amount_vs_avg", 0) > 0.5:
        reasons.append("Transaction amount significantly above user's historical average")
    
    if features.get("account_age_days", 365) < 30:
        reasons.append(f"Account is only {features.get('account_age_days', 0):.0f} days old")
    
    if ml_score > 0.7:
        reasons.append(f"ML anomaly model flagged this transaction (score: {ml_score:.3f})")

    return reasons
