"""
santhosh-fraudintel-real-time-platform
Main FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FraudIntel Platform starting up...")
    from backend.core.model_registry import ModelRegistry
    from backend.core.audit_logger import AuditLogger
    registry = ModelRegistry()
    await registry.initialize()
    app.state.model_registry = registry
    app.state.audit_logger = AuditLogger()
    logger.info("✅ All systems initialized.")
    yield
    logger.info("🛑 FraudIntel Platform shutting down...")


app = FastAPI(
    title="FraudIntel Real-Time Platform",
    description="Bank-grade fraud detection and digital identity anomaly system by Santhosh",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_latency_header(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# Routers
from backend.api.fraud_router import router as fraud_router
from backend.api.case_router import router as case_router
from backend.api.identity_router import router as identity_router
from backend.api.analytics_router import router as analytics_router
from backend.api.frontend_router import router as frontend_router

app.include_router(fraud_router, prefix="/api/v1/fraud", tags=["Fraud Scoring"])
app.include_router(case_router, prefix="/api/v1/cases", tags=["Case Management"])
app.include_router(identity_router, prefix="/api/v1/identity", tags=["Identity Graph"])
app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(frontend_router, tags=["Frontend"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "platform": "FraudIntel", "version": "1.0.0", "timestamp": time.time()}


@app.get("/api/v1/status")
async def system_status():
    return {
        "fraud_engine": "operational",
        "ml_models": "loaded",
        "identity_graph": "active",
        "streaming": "connected",
        "audit_log": "active",
    }
