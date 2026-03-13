import time
import uuid
from typing import Dict, Any, Tuple
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.schemas import InferenceRequest, GovernanceResult, EnforcementDecision, ExplanationOutput
from app.models.orm_models import InferenceEvent, AIModel, AuditLog
from app.modules.policy_engine import PolicyEngine
from app.modules.fairness_monitor import FairnessMonitor
from app.modules.explainability import ExplainabilityEngine
from app.modules.risk_scorer import RiskScorer
from app.modules.safety_scanner import SafetyScanner


class GovernanceService:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.fairness_monitor = FairnessMonitor()
        self.explainability_engine = ExplainabilityEngine()
        self.risk_scorer = RiskScorer()
        self.safety_scanner = SafetyScanner()

    async def evaluate_inference(
        self, 
        request: InferenceRequest, 
        db: AsyncSession, 
        model: AIModel,
        is_simulation: bool = False
    ) -> GovernanceResult:
        start_time = time.time()
        
        # Fairness evaluation
        raw_flags = self.fairness_monitor.evaluate(request.input_data, request.prediction, request.confidence)
        from app.models.schemas import FairnessFlag as FF
        fairness_flags = []
        for f in raw_flags:
            try:
                from app.core.config import settings
                fairness_flags.append(FF(
                    metric=f.get("metric", "unknown"),
                    group_a=f.get("group_a", "group_a"),
                    group_b=f.get("group_b", "group_b"),
                    disparity=float(f.get("disparity", 0.0)),
                    threshold=float(f.get("threshold", settings.FAIRNESS_DISPARITY_THRESHOLD)),
                    passed=bool(f.get("passed", False)),
                ))
            except Exception:
                pass

        inference_data = {
            "input_data": request.input_data, 
            "confidence": request.confidence, 
            "context": request.context or {}
        }
        
        # 1. Safety Scan for raw text (Shadow AI detection)
        if not request.input_data.get("toxicity_score") and not request.input_data.get("prompt_injection_score"):
            input_text = str(request.input_data.get("prompt", request.input_data.get("text", "")))
            output_text = str(request.prediction.get("content", request.prediction.get("text", "")))
            safety_results = self.safety_scanner.analyze_exchange(input_text, output_text)
            request.input_data.update(safety_results)
            inference_data["input_data"] = request.input_data

        flag_dicts = [f.model_dump() for f in fairness_flags]
        
        # 2. Policy Evaluation (Pass 1)
        policy_violations, _ = self.policy_engine.evaluate(inference_data, flag_dicts, 0.0)
        
        # 3. Risk Scoring (Derived from violations + flags)
        risk_score = self.risk_scorer.compute(request.confidence, flag_dicts, policy_violations, request.context or {})
        risk_level = self.risk_scorer.get_risk_level(risk_score)
        
        # 4. Final Enforcement (Check if Risk Score itself triggers a policy block)
        # We can now just check the risk-based rules manually or do a focused second pass if needed,
        # but let's keep it clean by updating the decision if risk is too high.
        violations_with_risk, final_decision = self.policy_engine.evaluate(inference_data, flag_dicts, risk_score)
        policy_violations = violations_with_risk # Ensure final response has the risk violation if triggered
        
        # Explainability
        domain = (request.context or {}).get("domain", "default")
        explanation = self.explainability_engine.explain(request.input_data, request.prediction, request.confidence, domain)

        inference_id = str(uuid.uuid4())
        processing_ms = round((time.time() - start_time) * 1000, 2)

        # Context metadata mapping
        context_metadata = {**(request.context or {}), "processing_ms": processing_ms}
        if is_simulation:
            context_metadata["source"] = "simulation"

        # Persist event
        event = InferenceEvent(
            id=inference_id,
            model_id=model.id,
            input_data=request.input_data,
            prediction=request.prediction,
            confidence=request.confidence,
            risk_score=risk_score,
            enforcement_decision=final_decision.value,
            fairness_flags=flag_dicts,
            policy_violations=policy_violations,
            explanation=explanation,
            context_metadata=context_metadata,
        )
        db.add(event)

        # Audit Logs
        audit_actor = request.model_id if not is_simulation else f"simulation/{domain}"
        db.add(AuditLog(
            event_type="inference_evaluated",
            entity_id=inference_id,
            entity_type="inference",
            actor=audit_actor,
            action=f"evaluated with decision={final_decision.value}",
            details={"risk_score": risk_score, "violations": len(policy_violations), "fairness_flags": len(fairness_flags), "scenario": domain if is_simulation else None},
            risk_level=risk_level.value,
        ))

        if final_decision == EnforcementDecision.BLOCK:
            db.add(AuditLog(
                event_type="model_blocked",
                entity_id=model.id,
                entity_type="ai_model",
                actor="governance_engine",
                action="blocked inference due to policy violation",
                details={"inference_id": inference_id, "violations": policy_violations[:3]},
                risk_level="critical"
            ))
        elif policy_violations:
            db.add(AuditLog(
                event_type="policy_violated", 
                entity_id=inference_id, 
                entity_type="inference",
                actor=audit_actor, 
                action="policy violation detected",
                details={"violations": policy_violations[:3]}, 
                risk_level=risk_level.value
            ))

        fairness_failed = [f for f in fairness_flags if not f.passed]
        if fairness_failed:
            db.add(AuditLog(
                event_type="fairness_issue_detected", 
                entity_id=inference_id, 
                entity_type="inference",
                actor=audit_actor, 
                action="fairness threshold exceeded",
                details={"flags": [f.model_dump() for f in fairness_failed]}, 
                risk_level="high"
            ))

        await db.commit()

        # Broadcast real-time update over WebSocket
        from app.services.websocket_manager import manager
        import asyncio
        asyncio.create_task(manager.broadcast({
            "type": "new_inference",
            "inference_id": inference_id,
            "risk_score": risk_score,
            "enforcement_decision": final_decision.value,
        }))

        return GovernanceResult(
            inference_id=inference_id,
            model_id=model.id,
            risk_score=risk_score,
            risk_level=risk_level,
            enforcement_decision=final_decision,
            fairness_flags=fairness_flags,
            policy_violations=policy_violations,
            risk_analysis={
                "composite_risk": risk_score,
                "risk_level": risk_level.value,
                "violation_count": len(policy_violations),
                "fairness_flags": len(fairness_flags),
                "shadow_ai_detected": context_metadata.get("shadow_ai_detected", False)
            },
            explanation=ExplanationOutput(**explanation),
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_ms,
        )

# Global service instance
governance_service = GovernanceService()
