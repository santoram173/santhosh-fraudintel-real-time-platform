"""
Streaming Pipeline (Mock Kafka / Redis Streams)
In production: replace MockStream with Kafka consumer/producer.
Processes transactions in real-time from a stream.
"""

import asyncio
import json
import time
import uuid
import random
import logging
import sys
from pathlib import Path
from typing import Callable, AsyncGenerator

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

logger = logging.getLogger(__name__)


class MockKafkaStream:
    """
    Simulates a Kafka stream with synthetic transaction events.
    Replace with `confluent_kafka.Consumer` in production.
    """

    def __init__(self, topic: str = "transactions", rate_per_second: float = 5.0):
        self.topic = topic
        self.rate = rate_per_second
        self._running = False

    async def produce_synthetic(self) -> AsyncGenerator[dict, None]:
        """Generate synthetic transaction events at configured rate."""
        from data.generate_sample_data import generate_transaction
        
        while self._running:
            is_fraud = random.random() < 0.05  # 5% fraud rate
            tx = generate_transaction(is_fraud)
            tx["stream_offset"] = int(time.time() * 1000)
            yield tx
            await asyncio.sleep(1.0 / self.rate)

    def start(self):
        self._running = True
        logger.info(f"📡 Mock Kafka stream started (topic={self.topic}, rate={self.rate}/s)")

    def stop(self):
        self._running = False
        logger.info("📡 Stream stopped")


class FraudStreamProcessor:
    """
    Consumes transaction stream and scores each transaction.
    Architecture:
      Stream → Feature Extraction → ML Scoring → Rule Engine → Output
    """

    def __init__(self):
        self.stream = MockKafkaStream(rate_per_second=3.0)
        self.processed = 0
        self.blocked = 0
        self.reviewed = 0
        self._output_handlers = []

    def register_output(self, handler: Callable):
        """Register a callback for scored transactions."""
        self._output_handlers.append(handler)

    async def process_transaction(self, tx: dict) -> dict:
        """Score a single transaction from stream."""
        from ml.features.feature_engineering import extract_features, features_to_vector
        from backend.rules.rule_engine import FraudRuleEngine
        from backend.core.fraud_scorer import RiskFusionEngine
        from ml.models.anomaly_model import FraudAnomalyModel

        rule_engine = FraudRuleEngine()
        risk_engine = RiskFusionEngine()

        features = extract_features(tx)
        feature_vector = features_to_vector(features)

        # Load model (would be cached in production)
        model = FraudAnomalyModel()
        if not model.load():
            ml_score = random.uniform(0.05, 0.3)
        else:
            ml_score = model.predict_score(feature_vector)

        rule_score, rule_reasons = rule_engine.evaluate(features, tx)
        risk_score = risk_engine.fuse(ml_score, rule_score, 0.1)
        decision = risk_engine.decide(risk_score)

        result = {
            "transaction_id": tx.get("transaction_id", str(uuid.uuid4())),
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "ml_score": round(ml_score, 4),
            "rule_score": round(rule_score, 2),
            "identity_risk": 0.1,
            "explanation": rule_reasons[:3],
            "stream_offset": tx.get("stream_offset"),
            "processed_at": time.time(),
        }

        self.processed += 1
        if decision == "BLOCK":
            self.blocked += 1
        elif decision == "REVIEW":
            self.reviewed += 1

        return result

    async def run(self, duration_seconds: float = 30.0):
        """Run stream processing for a duration."""
        self.stream.start()
        start = time.time()
        
        logger.info(f"🚀 Stream processor running for {duration_seconds}s...")
        
        async for tx in self.stream.produce_synthetic():
            if time.time() - start > duration_seconds:
                break
            
            result = await self.process_transaction(tx)
            
            for handler in self._output_handlers:
                await handler(result)
            
            if self.processed % 10 == 0:
                logger.info(
                    f"Processed: {self.processed} | "
                    f"Blocked: {self.blocked} | "
                    f"Reviewed: {self.reviewed}"
                )
        
        self.stream.stop()
        logger.info(f"✅ Stream processing complete. Total: {self.processed} transactions.")

    def get_stats(self) -> dict:
        return {
            "processed": self.processed,
            "blocked": self.blocked,
            "reviewed": self.reviewed,
            "approved": self.processed - self.blocked - self.reviewed,
            "block_rate": round(self.blocked / max(self.processed, 1) * 100, 2),
        }


async def demo_run():
    """Demo: run stream for 10 seconds and print results."""
    processor = FraudStreamProcessor()
    
    async def print_result(result: dict):
        icon = {"APPROVE": "✅", "REVIEW": "⚠️", "BLOCK": "🚨"}.get(result["decision"], "?")
        print(f"{icon} [{result['decision']:7}] risk={result['risk_score']:.1f} | {result['transaction_id'][:12]}")
    
    processor.register_output(print_result)
    await processor.run(duration_seconds=10)
    print(f"\n📊 Stats: {processor.get_stats()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_run())
