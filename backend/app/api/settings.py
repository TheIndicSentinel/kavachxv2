"""
Settings & Platform Configuration API
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings, save_thresholds
from app.core.auth import require_permission
from app.db.database import get_db

router = APIRouter()


# ── Minimum floor values — prevents accidental disabling of safety thresholds ──
_MIN_RISK_HIGH         = 0.50
_MIN_RISK_MEDIUM       = 0.10
_MIN_FAIRNESS_DISPARITY = 0.05
_MIN_CONFIDENCE_LOW    = 0.10


class ThresholdsModel(BaseModel):
    risk_high:           float
    risk_medium:         float
    fairness_disparity:  float
    confidence_low:      float

    @field_validator("risk_high")
    @classmethod
    def validate_risk_high(cls, v):
        if v < _MIN_RISK_HIGH:
            raise ValueError(f"risk_high must be >= {_MIN_RISK_HIGH} (received {v})")
        return v

    @field_validator("risk_medium")
    @classmethod
    def validate_risk_medium(cls, v):
        if v < _MIN_RISK_MEDIUM:
            raise ValueError(f"risk_medium must be >= {_MIN_RISK_MEDIUM} (received {v})")
        return v

    @field_validator("fairness_disparity")
    @classmethod
    def validate_fairness_disparity(cls, v):
        if v < _MIN_FAIRNESS_DISPARITY:
            raise ValueError(
                f"fairness_disparity must be >= {_MIN_FAIRNESS_DISPARITY} (received {v})"
            )
        return v

    @field_validator("confidence_low")
    @classmethod
    def validate_confidence_low(cls, v):
        if v < _MIN_CONFIDENCE_LOW:
            raise ValueError(
                f"confidence_low must be >= {_MIN_CONFIDENCE_LOW} (received {v})"
            )
        return v


@router.get("/thresholds", response_model=ThresholdsModel)
async def get_thresholds(current_user=Depends(require_permission("dashboard:read"))):
    return {
        "risk_high":          settings.RISK_SCORE_HIGH_THRESHOLD,
        "risk_medium":        settings.RISK_SCORE_MEDIUM_THRESHOLD,
        "fairness_disparity": settings.FAIRNESS_DISPARITY_THRESHOLD,
        "confidence_low":     settings.CONFIDENCE_LOW_THRESHOLD,
    }


@router.put("/thresholds", response_model=ThresholdsModel)
async def update_thresholds(
    data: ThresholdsModel,
    current_user=Depends(require_permission("policies:write")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update governance thresholds.

    Requires the `policies:write` permission (compliance_officer / super_admin).
    Every successful change is written to the immutable AuditLog so regulators
    can verify that thresholds were never silently lowered.
    """
    previous = {
        "risk_high":          settings.RISK_SCORE_HIGH_THRESHOLD,
        "risk_medium":        settings.RISK_SCORE_MEDIUM_THRESHOLD,
        "fairness_disparity": settings.FAIRNESS_DISPARITY_THRESHOLD,
        "confidence_low":     settings.CONFIDENCE_LOW_THRESHOLD,
    }

    save_thresholds(data.model_dump())

    # Write an immutable audit entry for every threshold change
    from app.models.orm_models import AuditLog
    import json as _json

    db.add(AuditLog(
        event_type="threshold_update",
        entity_id="kavachx_thresholds",
        entity_type="system_config",
        actor=current_user.get("email", current_user.get("name", "unknown")),
        action="update",
        details=_json.dumps({
            "previous": previous,
            "updated":  data.model_dump(),
        }),
    ))
    await db.commit()

    return {
        "risk_high":          settings.RISK_SCORE_HIGH_THRESHOLD,
        "risk_medium":        settings.RISK_SCORE_MEDIUM_THRESHOLD,
        "fairness_disparity": settings.FAIRNESS_DISPARITY_THRESHOLD,
        "confidence_low":     settings.CONFIDENCE_LOW_THRESHOLD,
    }
