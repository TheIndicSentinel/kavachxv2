"""ORM Models for KavachX database."""
from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, JSON, Integer, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from app.db.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class AIModel(Base):
    __tablename__ = "ai_models"
    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String)  # classification, regression, llm, etc.
    owner = Column(String)
    description = Column(Text)
    status = Column(String, default="active")  # active, suspended, archived
    registered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metadata_ = Column("metadata", JSON, default=dict)
    inferences = relationship("InferenceEvent", back_populates="model")


class GovernancePolicy(Base):
    __tablename__ = "governance_policies"
    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    policy_type = Column(String)  # fairness, safety, compliance, performance
    rules = Column(JSON)  # structured rule definitions
    severity = Column(String, default="medium")  # low, medium, high, critical
    enabled = Column(Boolean, default=True)
    jurisdiction = Column(String, default="IN")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class InferenceEvent(Base):
    __tablename__ = "inference_events"
    id = Column(String, primary_key=True, default=gen_uuid)
    model_id = Column(String, ForeignKey("ai_models.id"))
    model = relationship("AIModel", back_populates="inferences")
    input_data = Column(JSON)
    prediction = Column(JSON)
    confidence = Column(Float)
    risk_score = Column(Float)
    enforcement_decision = Column(String)  # PASS, ALERT, BLOCK, HUMAN_REVIEW
    fairness_flags = Column(JSON, default=list)
    policy_violations = Column(JSON, default=list)
    explanation = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    context_metadata = Column(JSON, default=dict)
    session_id = Column(String, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=gen_uuid)
    event_type = Column(String)  # inference_evaluated, policy_violated, model_blocked, etc.
    entity_id = Column(String)
    entity_type = Column(String)
    actor = Column(String)
    action = Column(String)
    details = Column(JSON)
    risk_level = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # Cryptographic-style integrity chain (hash of previous link + this payload)
    prev_hash = Column(String, nullable=True)
    chain_hash = Column(String, nullable=True, index=True)


class FairnessMetric(Base):
    __tablename__ = "fairness_metrics"
    id = Column(String, primary_key=True, default=gen_uuid)
    model_id = Column(String, ForeignKey("ai_models.id"))
    metric_name = Column(String)  # demographic_parity, equalized_odds, etc.
    group_a = Column(String)
    group_b = Column(String)
    value_a = Column(Float)
    value_b = Column(Float)
    disparity = Column(Float)
    passed = Column(Boolean)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
