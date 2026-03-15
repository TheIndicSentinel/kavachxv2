"""Pydantic schemas for KavachX API."""
from pydantic import BaseModel, Field, field_serializer
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum


class EnforcementDecision(str, Enum):
    PASS = "PASS"
    ALERT = "ALERT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --- Inference Request/Response ---

class InferenceRequest(BaseModel):
    model_id: Optional[str] = None
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = {}
    session_id: Optional[str] = None

class FairnessFlag(BaseModel):
    metric: str
    group_a: str
    group_b: str
    disparity: float
    threshold: float
    passed: bool

class ExplanationOutput(BaseModel):
    top_features: List[Dict[str, Any]]
    summary: str
    confidence_note: str
    reason: Optional[str] = None
    policy_triggered: Optional[str] = None

class GovernanceResult(BaseModel):
    inference_id: str
    model_id: str
    risk_score: float
    risk_level: RiskLevel
    enforcement_decision: EnforcementDecision
    fairness_flags: List[FairnessFlag]
    policy_violations: List[Dict[str, Any]]
    risk_analysis: Optional[Dict[str, Any]] = {}
    explanation: ExplanationOutput
    timestamp: datetime
    processing_time_ms: float

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")


# --- Policy Schemas ---

class PolicyRule(BaseModel):
    rule_id: str
    condition: str
    threshold: Optional[float] = None
    action: str  # alert, block, human_review
    message: str

class PolicyCreate(BaseModel):
    name: str
    description: str
    policy_type: str  # fairness, safety, compliance, performance
    rules: List[PolicyRule]
    severity: str = "medium"
    jurisdiction: str = "IN"

class PolicyOut(PolicyCreate):
    id: str
    enabled: bool
    created_at: datetime

    @field_serializer('created_at')
    def serialize_created_at(self, dt: datetime) -> str:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")


# --- Model Registration ---

class ModelRegister(BaseModel):
    name: str
    version: str
    model_type: str
    owner: str
    description: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}

class ModelOut(ModelRegister):
    id: str
    status: str
    registered_at: datetime

    @field_serializer('registered_at')
    def serialize_registered_at(self, dt: datetime) -> str:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")


# --- Dashboard ---

class DashboardStats(BaseModel):
    total_inferences: int
    blocked_count: int
    alert_count: int
    pass_rate: float
    avg_risk_score: float
    active_models: int
    policy_violations_today: int
    fairness_issues_detected: int

class RiskTrend(BaseModel):
    timestamp: str
    avg_risk_score: float
    inference_count: int

class ComplianceSummary(BaseModel):
    jurisdiction: str
    compliant_policies: int
    violated_policies: int
    compliance_rate: float
