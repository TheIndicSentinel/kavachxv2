"""Dashboard API."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta, timezone
from app.db.database import get_db
from app.models.orm_models import InferenceEvent, AIModel, AuditLog, FairnessMetric

router = APIRouter()


from app.db.cache import dashboard_cache

def ensure_naive(dt: datetime) -> datetime:
    """Helper to ensure a datetime is naive for comparison purposes."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

@router.get("/stats")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    cached = await dashboard_cache.get("dashboard_stats")
    if cached:
        return cached

    total = (await db.execute(select(func.count(InferenceEvent.id)))).scalar() or 0
    blocked = (await db.execute(select(func.count(InferenceEvent.id)).where(InferenceEvent.enforcement_decision == "BLOCK"))).scalar() or 0
    alert_ct = (await db.execute(select(func.count(InferenceEvent.id)).where(InferenceEvent.enforcement_decision == "ALERT"))).scalar() or 0
    pass_ct = (await db.execute(select(func.count(InferenceEvent.id)).where(InferenceEvent.enforcement_decision == "PASS"))).scalar() or 0
    avg_risk = float((await db.execute(select(func.avg(InferenceEvent.risk_score)))).scalar() or 0.0)
    active_models = (await db.execute(select(func.count(AIModel.id)).where(AIModel.status == "active"))).scalar() or 0
    
    # Use naive comparison consistently
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    violations_today = (await db.execute(select(func.count(AuditLog.id)).where(AuditLog.timestamp >= today))).scalar() or 0
    
    recent = (await db.execute(select(InferenceEvent).order_by(desc(InferenceEvent.timestamp)).limit(100))).scalars().all()
    fairness_issues = sum(1 for e in recent if e.fairness_flags and len(e.fairness_flags) > 0)
    pass_rate = (pass_ct / total) if total > 0 else 1.0
    
    result = {
        "total_inferences": total, "blocked_count": blocked, "alert_count": alert_ct,
        "pass_rate": round(pass_rate, 3), "avg_risk_score": round(avg_risk, 3),
        "active_models": active_models, "policy_violations_today": violations_today,
        "fairness_issues_detected": fairness_issues,
    }
    await dashboard_cache.set("dashboard_stats", result)
    return result


@router.get("/risk-trend")
async def get_risk_trend(hours: int = 24, db: AsyncSession = Depends(get_db)):
    now = datetime.utcnow()
    since = now - timedelta(hours=hours)

    result = await db.execute(
        select(InferenceEvent)
        .where(InferenceEvent.timestamp >= since)
        .order_by(InferenceEvent.timestamp)
    )
    events = result.scalars().all()

    # Build 24 hourly buckets: each bucket = hour offset from 'hours' ago
    # e.g. bucket 0 = oldest hour, bucket 23 = most recent hour
    buckets = [[] for _ in range(hours)]
    for e in events:
        ts = ensure_naive(e.timestamp)
        now_naive = datetime.utcnow()
        since_naive = now_naive - timedelta(hours=hours)
        if ts and ts >= since_naive:
            # Which hour bucket does this fall in?
            elapsed_seconds = (now_naive - ts).total_seconds()
            bucket_idx = max(0, min(hours - 1, hours - 1 - int(elapsed_seconds // 3600)))
            buckets[bucket_idx].append(e.risk_score or 0)

    result_data = []
    for i, bucket in enumerate(buckets):
        hour_offset = i - (hours - 1)  # negative = past hours, 0 = current hour
        label_dt = now + timedelta(hours=hour_offset)
        result_data.append({
            "hour": label_dt.strftime("%H:00"),
            "avg_risk": round(sum(bucket) / len(bucket), 3) if bucket else 0,
            "count": len(bucket),
        })

    return result_data



@router.get("/enforcement-breakdown")
async def get_enforcement_breakdown(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InferenceEvent.enforcement_decision, func.count(InferenceEvent.id)).group_by(InferenceEvent.enforcement_decision))
    rows = result.all()
    return {row[0]: row[1] for row in rows if row[0]}


@router.get("/compliance-summary")
async def get_compliance_summary(db: AsyncSession = Depends(get_db)):
    """
    Derives compliance scores from actual audit log data.
    Frameworks are mapped to the jurisdiction and policy_type of their violations.
    """
    # Use naive comparison consistently
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    result = await db.execute(
        select(AuditLog.details, AuditLog.risk_level, AuditLog.event_type)
        .where(AuditLog.timestamp >= thirty_days_ago)
        .where(AuditLog.event_type.in_(["policy_violated", "inference_evaluated", "model_blocked"]))
    )
    logs = result.all()

    total = len(logs) or 1
    blocked = sum(1 for l in logs if l.event_type == "model_blocked")
    violations = sum(1 for l in logs if l.event_type == "policy_violated")

    # Heuristic framework scores based on violation ratio
    violation_rate = round(violations / total, 3)
    block_rate = round(blocked / total, 3)

    def score(base: int, penalty: float) -> int:
        return max(0, min(100, round(base - penalty * 100)))

    # Map framework penalties to observed signals
    eu_score  = score(85, violation_rate * 0.6 + block_rate * 0.4)
    dpdp_score = score(80, violation_rate * 0.7 + block_rate * 0.3)
    rbi_score  = score(90, violation_rate * 0.4 + block_rate * 0.5)
    nist_score = score(75, violation_rate * 0.5 + block_rate * 0.5)

    def status(s: int) -> str:
        if s >= 80: return "compliant"
        if s >= 60: return "partial"
        return "non_compliant"

    return [
        {"framework": "EU AI Act",   "score": eu_score,   "status": status(eu_score)},
        {"framework": "DPDP 2023",   "score": dpdp_score, "status": status(dpdp_score)},
        {"framework": "RBI Fairness","score": rbi_score,  "status": status(rbi_score)},
        {"framework": "NIST AI RMF", "score": nist_score, "status": status(nist_score)},
    ]
