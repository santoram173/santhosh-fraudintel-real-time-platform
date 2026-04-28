"""
Test Suite for FraudIntel Platform
Tests: Feature Engineering, Rule Engine, ML Scoring, Risk Fusion, Identity Graph
"""

import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import time


class TestFeatureEngineering:
    def _tx(self, **kw):
        d = {"user_id":"tu001","amount":150.0,"currency":"USD","merchant_id":"m1",
             "merchant_category":"retail","country":"US","ip_address":"192.168.1.1",
             "device_id":"dev_001","device_type":"mobile","channel":"online","timestamp":time.time()}
        d.update(kw); return d

    def test_basic_features_extracted(self):
        from ml.features.feature_engineering import extract_features
        assert len(extract_features(self._tx())) >= 20

    def test_high_risk_country_flagged(self):
        from ml.features.feature_engineering import extract_features
        assert extract_features(self._tx(country="NG"))["is_high_risk_country"] == 1.0

    def test_normal_country_not_flagged(self):
        from ml.features.feature_engineering import extract_features
        assert extract_features(self._tx(country="US", user_id="tu002"))["is_high_risk_country"] == 0.0

    def test_vpn_ip_detected(self):
        from ml.features.feature_engineering import extract_features
        assert extract_features(self._tx(ip_address="10.0.0.50", user_id="tu003"))["is_vpn_proxy"] == 1.0

    def test_features_to_vector_shape(self):
        from ml.features.feature_engineering import extract_features, features_to_vector
        v = features_to_vector(extract_features(self._tx(user_id="tu004")))
        assert v.shape == (24,) and v.dtype == np.float32

    def test_large_amount_flagged(self):
        from ml.features.feature_engineering import extract_features
        assert extract_features(self._tx(amount=6000.0, user_id="tu005"))["is_large_amount"] == 1.0


class TestRuleEngine:
    def test_velocity_burst_triggers(self):
        from backend.rules.rule_engine import FraudRuleEngine
        score, reasons = FraudRuleEngine().evaluate(
            {"tx_count_1h":12,"velocity_spike":0.9,"is_new_device":0,"is_high_risk_country":0,
             "geo_mismatch":0,"distance_from_home":0,"is_vpn_proxy":0,"behavioral_drift":0,
             "unusual_hour":0,"merchant_risk":0,"session_anomaly":0,"is_round_amount":0,
             "account_age_days":365,"tx_count_24h":15},
            {"amount":200,"country":"US","merchant_category":"retail"}
        )
        assert score > 0
        assert any("velocity" in r.lower() for r in reasons)

    def test_new_account_high_value_triggers(self):
        from backend.rules.rule_engine import FraudRuleEngine
        score, reasons = FraudRuleEngine().evaluate(
            {"account_age_days":5,"is_new_device":0,"tx_count_1h":1,"is_high_risk_country":0,
             "is_vpn_proxy":0,"geo_mismatch":0,"distance_from_home":0,"is_round_amount":0,
             "unusual_hour":0,"merchant_risk":0,"behavioral_drift":0,"session_anomaly":0,
             "velocity_spike":0,"tx_count_24h":1,"ip_country_mismatch":0},
            {"amount":3000,"country":"US","merchant_category":"retail"}
        )
        assert any("new account" in r.lower() for r in reasons)

    def test_clean_transaction_low_score(self):
        from backend.rules.rule_engine import FraudRuleEngine
        base = {f:0.0 for f in ["tx_count_1h","velocity_spike","is_new_device","is_high_risk_country",
            "geo_mismatch","distance_from_home","is_vpn_proxy","behavioral_drift","unusual_hour",
            "merchant_risk","session_anomaly","is_round_amount","ip_country_mismatch","tx_count_24h"]}
        base["account_age_days"] = 365
        score, _ = FraudRuleEngine().evaluate(base, {"amount":50,"country":"US","merchant_category":"grocery"})
        assert score < 3.0


class TestRiskFusion:
    def test_low_all_signals_approves(self):
        from backend.core.fraud_scorer import RiskFusionEngine
        e = RiskFusionEngine()
        s = e.fuse(0.05, 0.0, 0.05)
        assert s < 3.5 and e.decide(s) == "APPROVE"

    def test_high_all_signals_blocks(self):
        from backend.core.fraud_scorer import RiskFusionEngine
        e = RiskFusionEngine()
        s = e.fuse(0.9, 8.0, 0.8)
        assert e.decide(s) in ("REVIEW","BLOCK")

    def test_score_capped_at_10(self):
        from backend.core.fraud_scorer import RiskFusionEngine
        assert RiskFusionEngine().fuse(1.0, 10.0, 1.0) <= 10.0

    def test_score_non_negative(self):
        from backend.core.fraud_scorer import RiskFusionEngine
        assert RiskFusionEngine().fuse(0.0, 0.0, 0.0) >= 0.0

    def test_mid_range_reviews(self):
        from backend.core.fraud_scorer import RiskFusionEngine
        e = RiskFusionEngine()
        assert e.decide(e.fuse(0.4, 4.0, 0.3)) in ("APPROVE","REVIEW")


class TestMLModel:
    @pytest.fixture(scope="class")
    def model(self):
        from ml.models.anomaly_model import FraudAnomalyModel
        from ml.training.train import generate_normal_transactions, generate_fraud_transactions
        m = FraudAnomalyModel()
        m.train(generate_normal_transactions(500), generate_fraud_transactions(50))
        return m

    def test_model_trains(self, model):
        assert model.is_trained

    def test_normal_low_score(self, model):
        from ml.training.train import generate_normal_transactions
        assert np.mean([model.predict_score(x) for x in generate_normal_transactions(10)]) < 0.6

    def test_fraud_high_score(self, model):
        from ml.training.train import generate_fraud_transactions
        assert np.mean([model.predict_score(x) for x in generate_fraud_transactions(10)]) > 0.4

    def test_score_in_range(self, model):
        s = model.predict_score(np.random.rand(24).astype(np.float32))
        assert 0.0 <= s <= 1.0


class TestIdentityGraph:
    def test_new_device_detected(self):
        from backend.core.identity_graph import IdentityGraph
        g = IdentityGraph()
        result = g.record_event("userA", f"dev_new_{uuid.uuid4().hex[:8]}", "1.2.3.4")
        assert result["is_new_device"] is True

    def test_known_device_not_new(self):
        from backend.core.identity_graph import IdentityGraph
        g = IdentityGraph()
        dev = f"dev_known_{uuid.uuid4().hex[:6]}"
        g.record_event("userB", dev, "1.2.3.5")
        g.record_event("userB", dev, "1.2.3.5")
        result = g.record_event("userB", dev, "1.2.3.5")
        assert result["is_new_device"] is False

    def test_shared_device_flagged(self):
        from backend.core.identity_graph import IdentityGraph
        g = IdentityGraph()
        dev = f"shared_{uuid.uuid4().hex[:6]}"
        for i in range(8):
            g.record_event(f"u_{i}", dev, f"1.2.3.{i+10}")
        result = g.record_event("u_final", dev, "1.2.3.99")
        assert result["device_risk"] > 0.3

    def test_composite_risk_bounded(self):
        from backend.core.identity_graph import IdentityGraph
        g = IdentityGraph()
        r = g.record_event("urisk", "dev_xyz", "10.0.0.1")
        assert 0.0 <= r["composite_risk"] <= 1.0

    def test_user_graph_snapshot(self):
        from backend.core.identity_graph import IdentityGraph
        g = IdentityGraph()
        g.record_event("usnap", "da", "5.5.5.5")
        g.record_event("usnap", "db", "5.5.5.6")
        snap = g.get_user_graph("usnap")
        assert snap["found"] is True and len(snap["devices"]) == 2


class TestPerformance:
    def test_feature_extraction_fast(self):
        from ml.features.feature_engineering import extract_features
        tx = {"user_id":"pu","amount":200,"currency":"USD","merchant_id":"m1",
              "merchant_category":"retail","country":"US","ip_address":"1.1.1.1",
              "device_id":"dp","timestamp":time.time()}
        start = time.perf_counter()
        for _ in range(100): extract_features(tx)
        assert (time.perf_counter()-start)*10 < 10, "Feature extraction too slow"

    def test_rule_engine_fast(self):
        from backend.rules.rule_engine import FraudRuleEngine
        e = FraudRuleEngine()
        f = {k:0.1 for k in ["tx_count_1h","velocity_spike","is_new_device","is_high_risk_country",
            "geo_mismatch","distance_from_home","is_vpn_proxy","behavioral_drift","unusual_hour",
            "merchant_risk","session_anomaly","is_round_amount","account_age_days","ip_country_mismatch","tx_count_24h"]}
        t = {"amount":100,"country":"US","merchant_category":"retail"}
        start = time.perf_counter()
        for _ in range(1000): e.evaluate(f, t)
        assert (time.perf_counter()-start) < 2, "Rule engine too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
