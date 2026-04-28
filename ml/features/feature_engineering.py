"""
Feature Engineering Layer
Transforms raw transaction data into ML-ready features.
"""

import hashlib
import math
import time
from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Simulated in-memory stores (replace with Redis/DB in production)
_transaction_history: Dict[str, list] = {}
_device_registry: Dict[str, dict] = {}
_ip_registry: Dict[str, dict] = {}
_user_profiles: Dict[str, dict] = {}

HIGH_RISK_COUNTRIES = {"NG", "RU", "CN", "BR", "PK", "IR", "KP"}
HIGH_RISK_MERCHANT_CATS = {"crypto", "gambling", "wire_transfer", "money_service"}
KNOWN_BAD_IPS = {"10.0.0.1", "172.16.0.1"}  # Placeholder blocklist


def extract_features(transaction: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract full feature vector from a transaction dict.
    Returns normalized features for ML model consumption.
    """
    user_id = transaction.get("user_id", "")
    device_id = transaction.get("device_id", "")
    ip_address = transaction.get("ip_address", "")
    amount = float(transaction.get("amount", 0))
    country = transaction.get("country", "US")
    merchant_category = transaction.get("merchant_category", "")
    timestamp = float(transaction.get("timestamp", time.time()))

    # --- Velocity Features ---
    velocity = _compute_velocity(user_id, amount, timestamp)

    # --- Amount Features ---
    amount_features = _compute_amount_features(user_id, amount)

    # --- Geo Features ---
    geo_features = _compute_geo_features(user_id, country, transaction.get("latitude"), transaction.get("longitude"))

    # --- Device Features ---
    device_features = _compute_device_features(user_id, device_id)

    # --- IP Features ---
    ip_features = _compute_ip_features(ip_address)

    # --- Behavioral Features ---
    behavioral = _compute_behavioral_features(user_id, timestamp)

    # --- Merchant Features ---
    merchant_risk = 1.0 if merchant_category.lower() in HIGH_RISK_MERCHANT_CATS else 0.0

    # --- Account Age ---
    account_age = _get_account_age(user_id)

    features = {
        # Velocity
        "tx_count_1h": float(velocity["count_1h"]),
        "tx_count_24h": float(velocity["count_24h"]),
        "tx_amount_1h": float(velocity["amount_1h"]),
        "tx_amount_24h": float(velocity["amount_24h"]),
        "velocity_spike": float(velocity["spike"]),
        # Amount
        "amount_normalized": amount_features["normalized"],
        "amount_vs_avg": amount_features["vs_avg"],
        "is_round_amount": amount_features["is_round"],
        "is_large_amount": 1.0 if amount > 5000 else 0.0,
        # Geo
        "is_high_risk_country": 1.0 if country in HIGH_RISK_COUNTRIES else 0.0,
        "geo_mismatch": float(geo_features["mismatch"]),
        "distance_from_home": float(geo_features["distance"]),
        # Device
        "is_new_device": float(device_features["is_new"]),
        "device_risk_score": float(device_features["risk"]),
        "device_tx_count": float(device_features["tx_count"]),
        # IP
        "ip_risk": float(ip_features["risk"]),
        "is_vpn_proxy": float(ip_features["is_vpn"]),
        "ip_country_mismatch": float(ip_features["country_mismatch"]),
        # Behavioral
        "unusual_hour": float(behavioral["unusual_hour"]),
        "behavioral_drift": float(behavioral["drift"]),
        "session_anomaly": float(behavioral["session_anomaly"]),
        # Merchant
        "merchant_risk": merchant_risk,
        # Account
        "account_age_days": float(account_age),
        "account_age_normalized": min(float(account_age) / 365.0, 1.0),
    }

    # Update history after extraction
    _update_history(user_id, device_id, ip_address, amount, timestamp)

    return features


def _compute_velocity(user_id: str, amount: float, timestamp: float) -> Dict:
    history = _transaction_history.get(user_id, [])
    now = timestamp
    count_1h = sum(1 for t in history if now - t["ts"] <= 3600)
    count_24h = sum(1 for t in history if now - t["ts"] <= 86400)
    amount_1h = sum(t["amount"] for t in history if now - t["ts"] <= 3600)
    amount_24h = sum(t["amount"] for t in history if now - t["ts"] <= 86400)
    
    # Spike detection: > 5 transactions in 1h is suspicious
    spike = 1.0 if count_1h > 5 else (count_1h / 5.0)
    
    return {
        "count_1h": min(count_1h, 20),
        "count_24h": min(count_24h, 100),
        "amount_1h": min(amount_1h, 50000),
        "amount_24h": min(amount_24h, 100000),
        "spike": spike,
    }


def _compute_amount_features(user_id: str, amount: float) -> Dict:
    history = _transaction_history.get(user_id, [])
    amounts = [t["amount"] for t in history[-50:]] if history else [amount]
    avg = np.mean(amounts) if amounts else amount
    std = np.std(amounts) if len(amounts) > 1 else 1.0
    
    normalized = min(amount / 10000.0, 1.0)
    vs_avg = min(abs(amount - avg) / (std + 1e-6) / 10.0, 1.0)  # z-score normalized
    is_round = 1.0 if amount % 100 == 0 and amount >= 500 else 0.0
    
    return {"normalized": normalized, "vs_avg": vs_avg, "is_round": is_round}


def _compute_geo_features(user_id: str, country: str, lat: Optional[float], lon: Optional[float]) -> Dict:
    profile = _user_profiles.get(user_id, {})
    home_country = profile.get("home_country", country)
    home_lat = profile.get("home_lat", lat or 0)
    home_lon = profile.get("home_lon", lon or 0)
    
    mismatch = 1.0 if country != home_country else 0.0
    
    distance = 0.0
    if lat and lon and home_lat and home_lon:
        distance = _haversine(lat, lon, home_lat, home_lon)
        distance = min(distance / 20000.0, 1.0)  # Normalize by max earth distance
    
    return {"mismatch": mismatch, "distance": distance}


def _compute_device_features(user_id: str, device_id: str) -> Dict:
    profile = _user_profiles.get(user_id, {})
    known_devices = profile.get("devices", [])
    device_info = _device_registry.get(device_id, {})
    
    is_new = 1.0 if device_id not in known_devices else 0.0
    tx_count = device_info.get("tx_count", 0)
    
    # Risk: new device + high tx count on it elsewhere = suspicious
    risk = is_new * 0.6 + (0.4 if tx_count > 20 and is_new else 0.0)
    
    return {"is_new": is_new, "risk": min(risk, 1.0), "tx_count": tx_count}


def _compute_ip_features(ip_address: str) -> Dict:
    is_vpn = 1.0 if ip_address.startswith("10.") or ip_address.startswith("172.") else 0.0
    ip_info = _ip_registry.get(ip_address, {})
    is_bad = 1.0 if ip_address in KNOWN_BAD_IPS else 0.0
    country_mismatch = float(ip_info.get("country_mismatch", 0))
    
    risk = min(is_vpn * 0.3 + is_bad * 0.7 + country_mismatch * 0.3, 1.0)
    return {"risk": risk, "is_vpn": is_vpn, "country_mismatch": country_mismatch}


def _compute_behavioral_features(user_id: str, timestamp: float) -> Dict:
    # Check if transaction is at unusual hour (2am - 5am local)
    hour = (timestamp % 86400) / 3600
    unusual_hour = 1.0 if 2 <= hour <= 5 else 0.0
    
    history = _transaction_history.get(user_id, [])
    
    # Behavioral drift: compare recent amounts to historical
    drift = 0.0
    if len(history) > 10:
        recent = [t["amount"] for t in history[-5:]]
        historical = [t["amount"] for t in history[-30:-5]]
        if historical:
            recent_avg = np.mean(recent)
            hist_avg = np.mean(historical)
            hist_std = np.std(historical) + 1e-6
            drift = min(abs(recent_avg - hist_avg) / hist_std / 5.0, 1.0)
    
    session_anomaly = 0.0
    if len(history) >= 2:
        last_ts = history[-1]["ts"]
        gap = abs(timestamp - last_ts)
        if gap < 30:  # Less than 30 seconds between transactions
            session_anomaly = 1.0
    
    return {"unusual_hour": unusual_hour, "drift": drift, "session_anomaly": session_anomaly}


def _get_account_age(user_id: str) -> float:
    profile = _user_profiles.get(user_id, {})
    created_at = profile.get("created_at", time.time() - 30 * 86400)  # Default: 30 days old
    return (time.time() - created_at) / 86400  # Days


def _update_history(user_id: str, device_id: str, ip_address: str, amount: float, timestamp: float):
    if user_id not in _transaction_history:
        _transaction_history[user_id] = []
    _transaction_history[user_id].append({"ts": timestamp, "amount": amount})
    # Keep last 100 transactions
    _transaction_history[user_id] = _transaction_history[user_id][-100:]
    
    # Update device registry
    if device_id not in _device_registry:
        _device_registry[device_id] = {"tx_count": 0, "first_seen": timestamp}
    _device_registry[device_id]["tx_count"] += 1
    _device_registry[device_id]["last_seen"] = timestamp
    
    # Update IP registry
    if ip_address not in _ip_registry:
        _ip_registry[ip_address] = {"tx_count": 0}
    _ip_registry[ip_address]["tx_count"] = _ip_registry[ip_address].get("tx_count", 0) + 1
    
    # Update user profile if new user
    if user_id not in _user_profiles:
        _user_profiles[user_id] = {
            "created_at": timestamp - np.random.uniform(30, 730) * 86400,
            "devices": [],
            "home_country": "US",
        }
    if device_id not in _user_profiles[user_id].get("devices", []):
        _user_profiles[user_id].setdefault("devices", []).append(device_id)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered numpy array for ML model."""
    FEATURE_ORDER = [
        "tx_count_1h", "tx_count_24h", "tx_amount_1h", "tx_amount_24h", "velocity_spike",
        "amount_normalized", "amount_vs_avg", "is_round_amount", "is_large_amount",
        "is_high_risk_country", "geo_mismatch", "distance_from_home",
        "is_new_device", "device_risk_score", "device_tx_count",
        "ip_risk", "is_vpn_proxy", "ip_country_mismatch",
        "unusual_hour", "behavioral_drift", "session_anomaly",
        "merchant_risk", "account_age_days", "account_age_normalized",
    ]
    return np.array([features.get(f, 0.0) for f in FEATURE_ORDER], dtype=np.float32)
