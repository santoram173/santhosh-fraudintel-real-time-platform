"""Pydantic schemas for all API request/response models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum
import time


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str
    amount: float = Field(..., gt=0)
    currency: str = "USD"
    merchant_id: str
    merchant_category: str
    country: str
    ip_address: str
    device_id: str
    device_type: str = "unknown"
    browser_fingerprint: Optional[str] = None
    session_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    channel: str = "online"
    timestamp: Optional[float] = None

    @validator("timestamp", pre=True, always=True)
    def set_timestamp(cls, v):
        return v or time.time()

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_abc123",
                "amount": 2500.00,
                "currency": "USD",
                "merchant_id": "merch_xyz789",
                "merchant_category": "electronics",
                "country": "US",
                "ip_address": "192.168.1.100",
                "device_id": "dev_abc123",
                "device_type": "mobile",
                "channel": "online"
            }
        }


class FraudScoreResponse(BaseModel):
    transaction_id: str
    decision: Decision
    risk_score: float = Field(..., ge=0, le=10, description="Fused risk score 0-10")
    ml_score: float = Field(..., ge=0, le=1)
    rule_score: float = Field(..., ge=0, le=10)
    identity_risk: float = Field(..., ge=0, le=1)
    explanation: List[str]
    features_used: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    model_version: str = "1.0.0"
    timestamp: float = Field(default_factory=time.time)


class CaseCreateRequest(BaseModel):
    transaction_id: str
    user_id: str
    risk_score: float
    decision: Decision
    notes: Optional[str] = None


class CaseUpdateRequest(BaseModel):
    status: CaseStatus
    analyst_id: str
    decision_override: Optional[Decision] = None
    notes: str


class Case(BaseModel):
    case_id: str
    transaction_id: str
    user_id: str
    risk_score: float
    decision: Decision
    status: CaseStatus
    created_at: float
    updated_at: float
    analyst_id: Optional[str] = None
    notes: Optional[str] = None
    audit_trail: List[Dict[str, Any]] = []


class IdentityGraphRequest(BaseModel):
    user_id: str
    device_id: str
    ip_address: str
    session_id: Optional[str] = None
    action: str = "login"


class IdentityRiskResponse(BaseModel):
    user_id: str
    device_risk: float
    ip_risk: float
    behavioral_risk: float
    composite_risk: float
    is_new_device: bool
    is_suspicious_ip: bool
    anomaly_signals: List[str]


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]


class AnalyticsResponse(BaseModel):
    total_transactions: int
    blocked: int
    reviewed: int
    approved: int
    avg_risk_score: float
    fraud_rate: float
    top_risk_countries: List[Dict]
    top_risk_merchants: List[Dict]
    hourly_volume: List[Dict]
