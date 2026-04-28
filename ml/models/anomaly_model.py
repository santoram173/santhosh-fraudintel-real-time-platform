"""
ML Anomaly Detection Model
Isolation Forest + XGBoost ensemble for fraud scoring.
"""

import os
import pickle
import time
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "saved_models"


class FraudAnomalyModel:
    """
    Ensemble anomaly detection model:
    - Isolation Forest for unsupervised anomaly scoring
    - XGBoost for supervised fraud classification
    - Final score = weighted ensemble
    """

    def __init__(self):
        self.isolation_forest = None
        self.xgboost_model = None
        self.scaler = None
        self.is_trained = False
        self.version = "1.0.0"
        self.feature_count = 24

    def train(self, X_normal: np.ndarray, X_fraud: Optional[np.ndarray] = None):
        """Train the ensemble model on transaction data."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        logger.info(f"Training anomaly model on {len(X_normal)} samples...")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_normal)

        # Isolation Forest (unsupervised)
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.05,  # Expect ~5% fraud rate
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)

        # XGBoost (supervised) if labeled fraud data available
        if X_fraud is not None:
            self._train_xgboost(X_normal, X_fraud)
        else:
            self._train_xgboost_synthetic(X_normal)

        self.is_trained = True
        logger.info("✅ Model training complete.")

    def _train_xgboost(self, X_normal: np.ndarray, X_fraud: np.ndarray):
        try:
            import xgboost as xgb
            X = np.vstack([X_normal, X_fraud])
            y = np.array([0] * len(X_normal) + [1] * len(X_fraud))
            X_scaled = self.scaler.transform(X)
            self.xgboost_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42,
            )
            self.xgboost_model.fit(X_scaled, y)
            logger.info("XGBoost trained successfully")
        except ImportError:
            logger.warning("XGBoost not available, using sklearn GradientBoosting")
            self._train_sklearn_gb(X_normal, X_fraud)

    def _train_sklearn_gb(self, X_normal: np.ndarray, X_fraud: np.ndarray):
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.vstack([X_normal, X_fraud])
        y = np.array([0] * len(X_normal) + [1] * len(X_fraud))
        X_scaled = self.scaler.transform(X)
        self.xgboost_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.xgboost_model.fit(X_scaled, y)

    def _train_xgboost_synthetic(self, X_normal: np.ndarray):
        """Generate synthetic fraud samples for supervised training."""
        n_fraud = max(int(len(X_normal) * 0.08), 50)
        
        # Synthetic fraud: high velocity, new device, high amount
        fraud_features = np.random.uniform(0.5, 1.0, (n_fraud, self.feature_count)).astype(np.float32)
        # Make them look fraudulent
        fraud_features[:, 0] = np.random.uniform(8, 20, n_fraud)  # high tx_count_1h
        fraud_features[:, 8] = 1.0   # is_large_amount
        fraud_features[:, 12] = 1.0  # is_new_device
        fraud_features[:, 9] = 1.0   # high_risk_country
        
        self._train_xgboost(X_normal, fraud_features)

    def predict_score(self, X: np.ndarray) -> float:
        """
        Returns fraud probability score 0-1.
        Higher = more fraudulent.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_reshaped = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X_reshaped)

        # Isolation Forest score: convert anomaly score to 0-1 probability
        if_score_raw = self.isolation_forest.decision_function(X_scaled)[0]
        # IF score is negative for anomalies; normalize to 0-1
        if_prob = 1.0 / (1.0 + np.exp(if_score_raw * 5))

        # XGBoost score
        xgb_prob = 0.5
        if self.xgboost_model is not None:
            xgb_prob = self.xgboost_model.predict_proba(X_scaled)[0][1]

        # Ensemble: 40% IF + 60% XGB
        ensemble_score = 0.4 * if_prob + 0.6 * xgb_prob

        return float(np.clip(ensemble_score, 0.0, 1.0))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction for streaming."""
        X_scaled = self.scaler.transform(X)
        if_scores = 1.0 / (1.0 + np.exp(self.isolation_forest.decision_function(X_scaled) * 5))
        if self.xgboost_model is not None:
            xgb_scores = self.xgboost_model.predict_proba(X_scaled)[:, 1]
            return np.clip(0.4 * if_scores + 0.6 * xgb_scores, 0, 1)
        return if_scores

    def save(self):
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH / "fraud_model.pkl", "wb") as f:
            pickle.dump({
                "isolation_forest": self.isolation_forest,
                "xgboost_model": self.xgboost_model,
                "scaler": self.scaler,
                "version": self.version,
            }, f)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load(self) -> bool:
        model_file = MODEL_PATH / "fraud_model.pkl"
        if not model_file.exists():
            return False
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        self.isolation_forest = data["isolation_forest"]
        self.xgboost_model = data["xgboost_model"]
        self.scaler = data["scaler"]
        self.version = data.get("version", "1.0.0")
        self.is_trained = True
        logger.info("Model loaded from disk.")
        return True
