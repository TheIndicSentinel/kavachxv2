"""
BASCG Startup Safety Checks
=============================
Called once from main.py lifespan before any service initialization.

Production guards — raise RuntimeError (fail fast) when:
  • BASCG_SIGNING_KEY_SEED_B64  is empty or still the placeholder default
  • SECRET_KEY                   is still the placeholder default
  • MOCK_TSA_SECRET              is still the placeholder default while
                                 SOVEREIGN_LEDGER_MODE is still "mock"
                                 (warn only — mock mode may be intentional in staging)

Development mode — emit WARNING for missing secrets but do NOT block startup.
This preserves zero-friction local dev while being strict in production.
"""

from __future__ import annotations

import base64
import logging

logger = logging.getLogger("kavachx.startup")

# Placeholder values that ship in .env.example — never valid in production
_PLACEHOLDER_SECRET_KEY       = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
_PLACEHOLDER_BASCG_SEED_B64   = ""   # empty means "not configured"
_PLACEHOLDER_MOCK_TSA_SECRET  = "CHANGE_ME_MOCK_TSA_SECRET_32CHARS"


def _is_production() -> bool:
    from app.core.config import settings
    return getattr(settings, "ENVIRONMENT", "development").lower() == "production"


def _check_bascg_signing_key() -> list[str]:
    """
    Returns a list of error strings (empty = OK).
    Production: BASCG_SIGNING_KEY_SEED_B64 must be set to a valid 32-byte base64 seed.
    """
    from app.core.config import settings
    errors: list[str] = []

    seed_b64 = getattr(settings, "BASCG_SIGNING_KEY_SEED_B64", "").strip()

    if not seed_b64:
        errors.append(
            "BASCG_SIGNING_KEY_SEED_B64 is not set.\n"
            "  Without a persistent signing key, every restart invalidates all NAEL\n"
            "  tokens and signed policy bundles.\n"
            "  Fix: run `python scripts/generate_dev_keys.py` and set the env var."
        )
        return errors

    # Validate: must decode to exactly 32 bytes
    try:
        raw = base64.b64decode(seed_b64)
        if len(raw) != 32:
            errors.append(
                f"BASCG_SIGNING_KEY_SEED_B64 decodes to {len(raw)} bytes — expected 32.\n"
                "  Fix: re-run `python scripts/generate_dev_keys.py` to generate a valid seed."
            )
    except Exception as exc:
        errors.append(
            f"BASCG_SIGNING_KEY_SEED_B64 is not valid base64: {exc}\n"
            "  Fix: re-run `python scripts/generate_dev_keys.py` to generate a valid seed."
        )

    return errors


def _check_secret_key() -> list[str]:
    """Returns a list of error strings (empty = OK)."""
    from app.core.config import settings
    errors: list[str] = []

    secret = getattr(settings, "SECRET_KEY", "").strip()
    if secret == _PLACEHOLDER_SECRET_KEY or not secret:
        errors.append(
            "SECRET_KEY is still the placeholder value.\n"
            "  Fix: set SECRET_KEY to the output of `openssl rand -hex 32`."
        )
    return errors


def _check_enforcement_flags() -> list[str]:
    """
    In production, NAEL and TEE enforcement must be enabled.
    Running production with these flags off means governance is advisory-only —
    a serious compliance gap under DPDP 2023 and the proposed BASCG framework.
    These are always warnings (never errors) because staging environments may
    intentionally keep them off during phased rollout.
    """
    from app.core.config import settings
    warnings_out: list[str] = []

    if not getattr(settings, "NAEL_ENFORCEMENT_ENABLED", True):
        warnings_out.append(
            "NAEL_ENFORCEMENT_ENABLED=False in a production environment. "
            "AI models without a valid NAEL license will NOT be blocked. "
            "Set NAEL_ENFORCEMENT_ENABLED=True or accept regulatory exposure."
        )

    if not getattr(settings, "TEE_ENFORCEMENT_ENABLED", True):
        warnings_out.append(
            "TEE_ENFORCEMENT_ENABLED=False in a production environment. "
            "Inference on compute nodes without a valid TEE clearance will NOT be blocked. "
            "Set TEE_ENFORCEMENT_ENABLED=True or accept regulatory exposure."
        )

    return warnings_out


def _check_mock_tsa_secret() -> list[str]:
    """
    In production with SOVEREIGN_LEDGER_MODE=mock, warn if MOCK_TSA_SECRET is
    still the placeholder. This is a WARNING, not an error — mock mode may be
    intentional in staging environments.
    """
    from app.core.config import settings
    warnings_out: list[str] = []

    mode       = getattr(settings, "SOVEREIGN_LEDGER_MODE", "mock").lower()
    mock_secret = getattr(settings, "MOCK_TSA_SECRET", "").strip()

    if mode == "mock" and mock_secret == _PLACEHOLDER_MOCK_TSA_SECRET:
        warnings_out.append(
            "MOCK_TSA_SECRET is still the placeholder value while "
            "SOVEREIGN_LEDGER_MODE=mock. "
            "Fix: set MOCK_TSA_SECRET to the output of `openssl rand -hex 32`, "
            "or switch to SOVEREIGN_LEDGER_MODE=rfc3161 for production anchoring."
        )
    return warnings_out


def run_production_checks() -> None:
    """
    Execute all startup safety checks.

    In production (ENVIRONMENT=production):
      - Any error → RuntimeError (app refuses to start)
    In development:
      - Errors are downgraded to WARNING (app starts but noisily)
    """
    prod = _is_production()

    errors:   list[str] = []
    warnings: list[str] = []

    errors.extend(_check_bascg_signing_key())
    errors.extend(_check_secret_key())
    warnings.extend(_check_mock_tsa_secret())
    if prod:
        warnings.extend(_check_enforcement_flags())

    for msg in warnings:
        logger.warning("BASCG startup warning: %s", msg)

    if not errors:
        if prod:
            logger.info("BASCG startup checks passed (production mode).")
        return

    if prod:
        lines = [
            "",
            "=" * 70,
            "  BASCG PRODUCTION STARTUP FAILURE",
            "  The following security checks failed:",
            "=" * 70,
        ]
        for i, err in enumerate(errors, 1):
            for line in err.splitlines():
                lines.append(f"  [{i}] {line}")
        lines += [
            "=" * 70,
            "  Set the missing environment variables and restart.",
            "=" * 70,
            "",
        ]
        raise RuntimeError("\n".join(lines))
    else:
        for msg in errors:
            logger.warning(
                "BASCG startup check FAILED (ignored in development): %s", msg
            )
