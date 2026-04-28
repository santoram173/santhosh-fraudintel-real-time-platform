"""Model Registry - manages ML model lifecycle."""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.fraud_model = None
        self._initialized = False

    async def initialize(self):
        """Load or train the fraud model."""
        # Add project root to path
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        
        from ml.models.anomaly_model import FraudAnomalyModel
        self.fraud_model = FraudAnomalyModel()
        
        if not self.fraud_model.load():
            logger.info("No saved model found. Training new model...")
            from ml.training.train import generate_normal_transactions, generate_fraud_transactions
            X_normal = generate_normal_transactions(5000)
            X_fraud = generate_fraud_transactions(400)
            self.fraud_model.train(X_normal, X_fraud)
            self.fraud_model.save()
        
        self._initialized = True
        logger.info("✅ Model registry initialized.")

    def get_fraud_model(self):
        if not self._initialized:
            raise RuntimeError("ModelRegistry not initialized")
        return self.fraud_model
