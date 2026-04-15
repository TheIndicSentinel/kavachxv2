"""Pydantic schemas for KavachX API."""
from pydantic import BaseModel, Field, field_serializer
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum


class EnforcementDecision(str, Enum):
    PASS             = "PASS"
    ALLOW_WITH_AUTH  = "ALLOW_WITH_AUTH"   # own-data request; requires authentication
    ALERT            = "ALERT"
    HUMAN_REVIEW     = "HUMAN_REVIEW"
    BLOCK            = "BLOCK"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --- Inference Request/Response ---

class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    # prediction and confidence are optional — real-time monitors (browser extension,
    # API probes) only have the prompt; governance engine synthesises defaults.
    prediction: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = Field(default=None, max_length=128)
    # BASCG P3: optional base64-encoded media bytes for synthetic media scan
    media_content_b64: Optional[str] = Field(default=None, description="Base64 media bytes for deepfake scan (BASCG P3)")
    media_content_type: Optional[str] = Field(default=None, description="MIME type hint for media_content_b64")

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_payload_size
        yield from super().__get_validators__()

    @classmethod
    def _validate_payload_size(cls, v):
        return v

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
    model_config = {"protected_namespaces": ()}
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
    ownership_context: Optional[Dict[str, Any]] = None  # set by ownership_detector

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
    model_config = {"protected_namespaces": ()}
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


# ── BASCG Phase 1: Sovereign Ledger Schemas ───────────────────────────────────

class LedgerAnchorOut(BaseModel):
    """Summary of a LedgerAnchor returned by the API."""
    id: str
    batch_start_log_id: Optional[str]
    batch_end_log_id: Optional[str]
    log_count: int
    merkle_root: str
    tsa_provider: Optional[str]
    tsa_serial: Optional[str]
    tsa_timestamp: Optional[str]    # ISO-8601 UTC
    anchor_status: str              # pending | anchored | failed
    error_message: Optional[str]
    created_at: str
    anchored_at: Optional[str]


class MerkleProofStep(BaseModel):
    """One sibling node in a Merkle proof path."""
    direction: str   # "left" | "right"
    hash: str        # hex SHA-256 of the sibling node


class MerkleProofOut(BaseModel):
    """
    Self-contained Merkle proof package for a single AuditLog entry.

    Verification algorithm:
        current = SHA256(bytes.fromhex(chain_hash))
        for step in proof:
            sibling = bytes.fromhex(step.hash)
            current_b = bytes.fromhex(current)
            if step.direction == "left":
                current = SHA256(sibling + current_b).hex()
            else:
                current = SHA256(current_b + sibling).hex()
        assert current == merkle_root  # tamper-evident

    The tsa_token_b64 independently witnesses that merkle_root existed at
    tsa_timestamp, satisfying IT Act S.65B electronic evidence requirements.
    """
    anchor_id: str
    merkle_root: str
    leaf_index: int
    leaf_hash: str              # SHA256(chain_hash) — the Merkle leaf node
    proof: List[MerkleProofStep]
    log_count: int              # total logs in this anchor batch
    tsa_provider: Optional[str]
    tsa_token_b64: Optional[str]
    tsa_timestamp: Optional[str]
    anchor_status: str
    verification_hint: str      # human-readable verification instructions
