"""
Model Training Script
Generates synthetic data and trains the fraud detection model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import logging
from ml.models.anomaly_model import FraudAnomalyModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COUNT = 24


def generate_normal_transactions(n: int = 5000) -> np.ndarray:
    """Generate synthetic normal (legitimate) transaction features."""
    np.random.seed(42)
    X = np.zeros((n, FEATURE_COUNT), dtype=np.float32)

    # tx_count_1h: mostly low
    X[:, 0] = np.random.exponential(1.5, n).clip(0, 20)
    # tx_count_24h
    X[:, 1] = np.random.exponential(5, n).clip(0, 100)
    # tx_amount_1h
    X[:, 2] = np.random.exponential(500, n).clip(0, 50000)
    # tx_amount_24h
    X[:, 3] = np.random.exponential(1500, n).clip(0, 100000)
    # velocity_spike
    X[:, 4] = np.random.beta(1, 5, n)
    # amount_normalized
    X[:, 5] = np.random.exponential(0.1, n).clip(0, 1)
    # amount_vs_avg
    X[:, 6] = np.random.beta(1, 4, n)
    # is_round_amount
    X[:, 7] = np.random.binomial(1, 0.05, n).astype(float)
    # is_large_amount
    X[:, 8] = np.random.binomial(1, 0.03, n).astype(float)
    # is_high_risk_country
    X[:, 9] = np.random.binomial(1, 0.05, n).astype(float)
    # geo_mismatch
    X[:, 10] = np.random.beta(1, 8, n)
    # distance_from_home
    X[:, 11] = np.random.exponential(0.05, n).clip(0, 1)
    # is_new_device
    X[:, 12] = np.random.binomial(1, 0.1, n).astype(float)
    # device_risk_score
    X[:, 13] = np.random.beta(1, 6, n)
    # device_tx_count
    X[:, 14] = np.random.exponential(10, n).clip(0, 200)
    # ip_risk
    X[:, 15] = np.random.beta(1, 7, n)
    # is_vpn_proxy
    X[:, 16] = np.random.binomial(1, 0.08, n).astype(float)
    # ip_country_mismatch
    X[:, 17] = np.random.binomial(1, 0.06, n).astype(float)
    # unusual_hour
    X[:, 18] = np.random.binomial(1, 0.1, n).astype(float)
    # behavioral_drift
    X[:, 19] = np.random.beta(1, 8, n)
    # session_anomaly
    X[:, 20] = np.random.binomial(1, 0.05, n).astype(float)
    # merchant_risk
    X[:, 21] = np.random.binomial(1, 0.04, n).astype(float)
    # account_age_days
    X[:, 22] = np.random.exponential(200, n).clip(1, 3650)
    # account_age_normalized
    X[:, 23] = np.clip(X[:, 22] / 365.0, 0, 1)

    return X


def generate_fraud_transactions(n: int = 400) -> np.ndarray:
    """Generate synthetic fraud transaction features."""
    np.random.seed(99)
    X = np.zeros((n, FEATURE_COUNT), dtype=np.float32)

    X[:, 0] = np.random.uniform(8, 20, n)      # high velocity
    X[:, 1] = np.random.uniform(20, 100, n)
    X[:, 2] = np.random.uniform(3000, 50000, n)
    X[:, 3] = np.random.uniform(10000, 100000, n)
    X[:, 4] = np.random.uniform(0.6, 1.0, n)   # velocity spike
    X[:, 5] = np.random.uniform(0.5, 1.0, n)   # large amounts
    X[:, 6] = np.random.uniform(0.5, 1.0, n)   # unusual vs avg
    X[:, 7] = np.random.binomial(1, 0.4, n).astype(float)
    X[:, 8] = np.random.binomial(1, 0.6, n).astype(float)   # large amount
    X[:, 9] = np.random.binomial(1, 0.4, n).astype(float)   # high risk country
    X[:, 10] = np.random.uniform(0.5, 1.0, n)  # geo mismatch
    X[:, 11] = np.random.uniform(0.3, 1.0, n)  # far from home
    X[:, 12] = np.random.binomial(1, 0.7, n).astype(float)  # new device
    X[:, 13] = np.random.uniform(0.5, 1.0, n)
    X[:, 14] = np.random.uniform(0, 5, n)
    X[:, 15] = np.random.uniform(0.5, 1.0, n)  # high ip risk
    X[:, 16] = np.random.binomial(1, 0.5, n).astype(float)  # vpn
    X[:, 17] = np.random.binomial(1, 0.5, n).astype(float)
    X[:, 18] = np.random.binomial(1, 0.4, n).astype(float)  # unusual hour
    X[:, 19] = np.random.uniform(0.4, 1.0, n)  # behavioral drift
    X[:, 20] = np.random.binomial(1, 0.4, n).astype(float)
    X[:, 21] = np.random.binomial(1, 0.3, n).astype(float)
    X[:, 22] = np.random.uniform(1, 30, n)     # new accounts
    X[:, 23] = np.clip(X[:, 22] / 365.0, 0, 1)

    return X


def train_and_save():
    logger.info("🧠 Starting model training pipeline...")
    
    X_normal = generate_normal_transactions(5000)
    X_fraud = generate_fraud_transactions(400)
    
    logger.info(f"Normal samples: {len(X_normal)}, Fraud samples: {len(X_fraud)}")
    
    model = FraudAnomalyModel()
    model.train(X_normal, X_fraud)
    model.save()
    
    # Quick validation
    test_normal = generate_normal_transactions(100)
    test_fraud = generate_fraud_transactions(50)
    
    normal_scores = [model.predict_score(x) for x in test_normal[:20]]
    fraud_scores = [model.predict_score(x) for x in test_fraud[:20]]
    
    logger.info(f"Avg score (normal): {np.mean(normal_scores):.3f}")
    logger.info(f"Avg score (fraud):  {np.mean(fraud_scores):.3f}")
    logger.info("✅ Model training and validation complete!")
    
    return model


if __name__ == "__main__":
    train_and_save()
