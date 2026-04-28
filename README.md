# 🛡️ FraudIntel — Real-Time Fraud Intelligence Platform

**`santhosh-fraudintel-real-time-platform`**

> Bank-grade fraud detection and digital identity anomaly system with sub-100ms scoring, ML anomaly detection, rule-based engine, identity graph, and analyst case management.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FraudIntel Platform                              │
│                                                                     │
│  ┌──────────┐    ┌────────────────────────────────────────────┐    │
│  │ Clients  │    │            FastAPI Backend                  │    │
│  │          │───▶│  /fraud/score  /cases  /identity           │    │
│  │ Web UI   │    │  /analytics    /audit                      │    │
│  │ API      │    └──────────────────┬─────────────────────────┘    │
│  │ Stream   │                       │                              │
│  └──────────┘          ┌────────────▼────────────┐                │
│                         │   Request Pipeline      │                │
│                         │                         │                │
│                    ┌────▼────┐  ┌──────────┐      │                │
│                    │Feature  │  │  Rule    │      │                │
│                    │Engineer │  │  Engine  │      │                │
│                    │  Layer  │  │ (12 rules│      │                │
│                    └────┬────┘  └────┬─────┘      │                │
│                         │            │             │                │
│                    ┌────▼────┐  ┌────▼─────┐      │                │
│                    │Isolation│  │ Identity │      │                │
│                    │Forest + │  │  Graph   │      │                │
│                    │XGBoost  │  │ (User→   │      │                │
│                    │Ensemble │  │  Device  │      │                │
│                    └────┬────┘  │  →IP)    │      │                │
│                         │       └────┬─────┘      │                │
│                         └──────┬─────┘             │                │
│                          ┌─────▼──────┐            │                │
│                          │Risk Fusion │            │                │
│                          │  Engine    │            │                │
│                          │ 0–10 Score │            │                │
│                          └─────┬──────┘            │                │
│                     ┌──────────▼──────────┐        │                │
│              ┌──────┤   Decision Engine   ├──────┐ │                │
│              │      └─────────────────────┘      │ │                │
│           APPROVE       REVIEW                BLOCK│                │
│              │             │                    │  │                │
│              │      ┌──────▼──────┐             │  │                │
│              │      │   Case      │             │  │                │
│              │      │  Manager    │◀────────────┘  │                │
│              │      │ (Analyst WF)│                │                │
│              │      └──────┬──────┘                │                │
│              │             │                       │                │
│              └──────┬──────┘                       │                │
│                     │                              │                │
│              ┌──────▼──────┐                       │                │
│              │ Audit Logger│ (Immutable log)        │                │
│              └─────────────┘                       │                │
└─────────────────────────────────────────────────────────────────────┘

Decision Logic:
  risk_score < 3.5  →  APPROVE  ✅
  risk_score < 6.5  →  REVIEW   ⚠️
  risk_score ≥ 6.5  →  BLOCK    🚨

Risk Fusion Formula:
  risk_score = (0.45 × ml_score + 0.35 × rule_score/10 + 0.20 × identity_risk) × 10
```

---

## Quick Start

### Option 1: Local (Recommended for development)

```bash
# 1. Clone / unzip the repository
cd santhosh-fraudintel-real-time-platform

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample data
python data/generate_sample_data.py

# 5. Train the ML model
python -m ml.training.train

# 6. Start the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Open dashboard
open http://localhost:8000
# API docs: http://localhost:8000/api/docs
```

### Option 2: Docker (One-click)

```bash
# From project root
cd docker
docker-compose up --build

# API available at: http://localhost:8000
# Docs at:         http://localhost:8000/api/docs
```

---

## API Reference

### Score a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/fraud/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_0042",
    "amount": 2500.00,
    "currency": "USD",
    "merchant_id": "merch_001",
    "merchant_category": "crypto",
    "country": "NG",
    "ip_address": "10.0.0.55",
    "device_id": "dev_new_xyz",
    "device_type": "mobile",
    "channel": "online"
  }'
```

**Response:**
```json
{
  "transaction_id": "3f7a-...",
  "decision": "BLOCK",
  "risk_score": 8.4,
  "ml_score": 0.8721,
  "rule_score": 7.5,
  "identity_risk": 0.65,
  "explanation": [
    "[RULE] New device with high transaction amount: $2500.00",
    "[RULE] Transaction from high-risk country: NG",
    "[RULE] VPN/Proxy IP detected",
    "[ML] Transaction initiated from unrecognized device",
    "[IDENTITY] New device detected for this user"
  ],
  "processing_time_ms": 12.4,
  "model_version": "1.0.0"
}
```

---

## Project Structure

```
fraudintel/
├── backend/                    # FastAPI application
│   ├── api/                   # Route handlers
│   │   ├── fraud_router.py   # /fraud/* endpoints
│   │   ├── case_router.py    # /cases/* endpoints
│   │   ├── identity_router.py# /identity/* endpoints
│   │   ├── analytics_router.py
│   │   └── frontend_router.py
│   ├── core/                  # Core business logic
│   │   ├── fraud_scorer.py   # Risk fusion engine
│   │   ├── model_registry.py # ML model lifecycle
│   │   ├── audit_logger.py   # Immutable audit log
│   │   └── identity_graph.py # User→Device→IP graph
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response schemas
│   ├── rules/
│   │   └── rule_engine.py    # 12 fraud detection rules
│   ├── explainability/
│   │   └── explainer.py      # Human-readable explanations
│   └── main.py               # FastAPI app + lifespan
├── ml/                        # Machine learning layer
│   ├── features/
│   │   └── feature_engineering.py  # 24-feature extraction
│   ├── models/
│   │   └── anomaly_model.py  # IsolationForest + XGBoost ensemble
│   └── training/
│       └── train.py          # Model training pipeline
├── streaming/
│   └── stream_processor.py   # Mock Kafka stream processor
├── frontend/
│   └── index.html            # Real-time dashboard
├── data/
│   ├── generate_sample_data.py
│   └── audit_log.jsonl       # Auto-generated audit trail
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   └── postman_collection.json
├── tests/
│   └── test_fraud_engine.py  # Full test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_fraud_engine.py::TestMLModel -v
pytest tests/test_fraud_engine.py::TestPerformance -v
```

---

## Features

| Feature | Implementation |
|---|---|
| Real-time scoring | FastAPI async + sub-100ms target |
| ML anomaly detection | Isolation Forest + XGBoost ensemble |
| Rule engine | 12 configurable fraud rules |
| Risk fusion | Weighted scoring (ML 45% + Rules 35% + Identity 20%) |
| Identity graph | User→Device→IP relationship model |
| Explainable AI | Human-readable reason codes per decision |
| Case management | OPEN/IN_REVIEW/CLOSED/ESCALATED workflow |
| Audit logging | Immutable JSONL audit trail |
| Batch scoring | Up to 100 transactions per request |
| Mock streaming | Kafka-compatible stream processor |
| Dashboard | Real-time HTML/JS fraud operations center |

---

## ML Model Details

### Feature Engineering (24 features)
- **Velocity**: tx count/amount in 1h and 24h windows
- **Amount**: normalized, vs user average, round-amount detection
- **Geo**: high-risk country, geo mismatch, distance from home
- **Device**: new device detection, device risk score, shared-device detection
- **IP**: VPN/proxy detection, IP reputation, country mismatch
- **Behavioral**: unusual hour, behavioral drift, session anomaly
- **Merchant**: high-risk category detection
- **Account**: account age (days)

### Ensemble Architecture
```
Feature Vector (24-dim)
        │
   ┌────┴────┐
   │         │
Isolation  XGBoost
Forest     Classifier
(40%)      (60%)
   │         │
   └────┬────┘
        │
  Ensemble Score
    (0.0 – 1.0)
```

---

## Extending to Production

| Component | Current | Production Upgrade |
|---|---|---|
| Storage | In-memory dicts | PostgreSQL + Redis |
| Streaming | Mock generator | Apache Kafka |
| ML serving | In-process | Triton / BentoML |
| Identity Graph | In-memory | Neo4j / TigerGraph |
| Audit Log | JSONL file | Kafka + S3/BigQuery |
| Auth | None | JWT + API keys |
| Monitoring | Logs | Prometheus + Grafana |

---

## License

MIT — Built by Santhosh as a POC/reference architecture.
