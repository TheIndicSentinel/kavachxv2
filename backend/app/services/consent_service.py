from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timezone
from app.models.orm_models import ConsentRecord


class ConsentService:
    """Verifies data principal consent in accordance with India DPDP 2023."""

    async def verify_consent(self, db: AsyncSession, principal_id: str, purpose: str) -> bool:
        """
        Check if valid, non-expired consent exists for the principal and purpose.
        Also checks the in-memory revocation set maintained by ConnectionManager.
        """
        if not principal_id or not purpose:
            return False

        # Fast path: already revoked in-memory (DPDP §6 — real-time propagation)
        from app.services.websocket_manager import manager
        if manager.is_revoked(principal_id):
            return False

        result = await db.execute(
            select(ConsentRecord)
            .where(and_(
                ConsentRecord.data_principal_id == principal_id,
                ConsentRecord.purpose == purpose,
                ConsentRecord.consent_given == True,
            ))
        )
        record = result.scalars().first()

        if not record:
            return False

        # Check expiry
        if record.expires_at and record.expires_at < datetime.now(timezone.utc):
            return False

        return True

    async def record_consent(
        self,
        db: AsyncSession,
        principal_id: str,
        purpose: str,
        given: bool = True,
        expiry: datetime = None,
    ) -> ConsentRecord:
        """
        Creates or updates a consent record.

        When given=False (revocation), broadcasts the revocation to all active
        WebSocket sessions so in-flight inference requests can be aborted
        immediately — as required by DPDP 2023 §6.
        """
        result = await db.execute(
            select(ConsentRecord)
            .where(and_(
                ConsentRecord.data_principal_id == principal_id,
                ConsentRecord.purpose == purpose,
            ))
        )
        record = result.scalars().first()

        if record:
            record.consent_given = given
            record.collected_at  = datetime.now(timezone.utc)
            record.expires_at    = expiry
        else:
            record = ConsentRecord(
                data_principal_id=principal_id,
                purpose=purpose,
                consent_given=given,
                expires_at=expiry,
            )
            db.add(record)

        await db.commit()

        # Propagate revocation to active sessions via WebSocket broadcast
        if not given:
            from app.services.websocket_manager import manager
            await manager.broadcast_consent_revocation(principal_id)

        return record


consent_service = ConsentService()
