"""KavachX Configuration Settings"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "KavachX Governance Engine"
    DATABASE_URL: str = "sqlite+aiosqlite:///./kavachx.db"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173,http://127.0.0.1:3000"
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    ENVIRONMENT: str = "development"
    PORT: int = 8001
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours
    MAX_CONNECTION_POOL: int = 10

    # Governance thresholds
    RISK_SCORE_HIGH_THRESHOLD: float = 0.60
    RISK_SCORE_MEDIUM_THRESHOLD: float = 0.20
    FAIRNESS_DISPARITY_THRESHOLD: float = 0.20
    CONFIDENCE_LOW_THRESHOLD: float = 0.55

    # Security
    EXPOSE_DEMO_PASSWORDS: bool = False  # set True only in ENVIRONMENT=development contexts
    MAX_REQUEST_SIZE_KB: int = 512  # reject inference payloads larger than this
    LOGIN_MAX_FAILURES: int = 10  # rate limit per IP per minute

    # Audit
    AUDIT_PROBE_PROBABILITY: float = 0.05  # random governance sampling probability

    # Display
    DISPLAY_TIMEZONE: str = "Asia/Kolkata"  # IST for India-first deployments

    # ── BASCG Phase 1: Sovereign Ledger Sync ────────────────────────────────
    # Layer 4 (Forensic) — Merkle anchoring for IT Act S.65B legal admissibility
    SOVEREIGN_LEDGER_ENABLED: bool = True
    # "mock"    → local HMAC-signed token, no network (dev/CI)
    # "rfc3161" → live RFC 3161 HTTP request to TSA_URL (production)
    SOVEREIGN_LEDGER_MODE: str = "mock"
    LEDGER_ANCHOR_INTERVAL_MINUTES: int = 5
    LEDGER_ANCHOR_MIN_BATCH_SIZE: int = 1   # set higher in prod to reduce TSA calls
    TSA_URL: str = "https://freetsa.org/tsr"  # FreeTSA: free public RFC 3161 endpoint
    MOCK_TSA_SECRET: str = "CHANGE_ME_MOCK_TSA_SECRET_32CHARS"  # HMAC key for mock mode

    # ── BASCG Phase 1: Cryptographic Signing (P0 + P1) ──────────────────────
    # Shared Ed25519 seed for dev/local signing of PolicyBundles and NAEL tokens.
    # Generate with: python scripts/generate_dev_keys.py
    # Production: replace with HSM-backed key (AWS KMS, Google Cloud HSM).
    # Value: base64-encoded 32-byte seed.  Leave empty → ephemeral key each restart.
    BASCG_SIGNING_KEY_SEED_B64: str = ""
    # JSON dict of additional trusted issuer public keys for production:
    # '{"MeitY-BASCG-v1": "<base64_pub>", "RBI-AI-Gov": "<base64_pub>"}'
    BASCG_TRUSTED_PUBLIC_KEYS_JSON: str = "{}"
    # When True, inferences on models without a valid NAEL license are BLOCKED.
    # Set False during onboarding / migration period.
    NAEL_ENFORCEMENT_ENABLED: bool = False
    # TEE attestation mode: "mock" | "aws-nitro"
    TEE_ATTESTATION_MODE: str = "mock"
    # When True, inferences on compute nodes without a valid TEE clearance are BLOCKED.
    # Mirror of NAEL_ENFORCEMENT_ENABLED — flip after attestation is bootstrapped.
    TEE_ENFORCEMENT_ENABLED: bool = False
    # Local Bridge: auto-generate a mock attestation clearance for any inference
    # session that has no existing clearance, when ENVIRONMENT=development.
    # Production: set False — only manually attested nodes may run inference.
    TEE_AUTO_ATTEST_DEV: bool = True
    # How long a TEE clearance stays valid after a successful attestation (minutes).
    # Override here to share with the service without hardcoding it in two places.
    TEE_CLEARANCE_TTL_MINUTES: int = 60

    # ── BASCG Phase 3: Synthetic Media Shield ───────────────────────────────
    # Detector mode: "mock" (deterministic hash-based) | "api" (HTTP detector)
    SYNTHETIC_MEDIA_MODE: str = "mock"       # "mock" | "api" | "onnx"
    SYNTHETIC_MEDIA_API_URL: str = ""        # Primary detector endpoint (required when mode=api)
    SYNTHETIC_MEDIA_API_KEY: str = ""        # API key for primary detector
    # Secondary / bridge detector — tried if primary API fails before dropping to local heuristics.
    # Set to any HF model ID or API endpoint; leave empty to disable the bridge tier.
    # Example: "openai/siglip-base-patch16-224"  (general vision model as bridge)
    SYNTHETIC_MEDIA_API_URL_2: str = ""      # Bridge detector endpoint (optional)
    SYNTHETIC_MEDIA_API_KEY_2: str = ""      # API key for bridge detector
    # ── ONNX local model (T2-C) ───────────────────────────────────────────────
    SYNTHETIC_MEDIA_ONNX_MODEL_PATH: str = ""     # path to .onnx deepfake classifier
    SYNTHETIC_MEDIA_ONNX_INPUT_NAME: str = "input"  # ONNX input node name
    SYNTHETIC_MEDIA_ONNX_INPUT_SIZE: int = 224    # image resize target (pixels)
    SYNTHETIC_MEDIA_CONFIDENCE_THRESHOLD: float = 0.45  # flag above this confidence (45%)
    # Election Protection Mode — escalates synthetic detections to ECI integrity bus
    ELECTION_PROTECTION_ENABLED: bool = False
    ELECTION_PROTECTION_STATE: str = ""     # 2-letter state code e.g. "MH"

    # ── ECI Election Integrity Webhook ──────────────────────────────────────
    # "stub" → log-only, no network (default dev/CI)
    # "http" → sign payload with BASCG Ed25519 key and POST to ECI_WEBHOOK_URL
    ECI_WEBHOOK_MODE: str = "stub"
    ECI_WEBHOOK_URL: str = ""           # required in http mode
    ECI_WEBHOOK_API_KEY: str = ""       # Bearer token for ECI endpoint
    ECI_WEBHOOK_TIMEOUT_SECONDS: int = 10

    # ── BASCG Phase 2b: Blockchain TSA (EVM permissioned anchor) ────────────
    # Used when SOVEREIGN_LEDGER_MODE="blockchain"
    # RPC endpoint for permissioned Ethereum PoA / Hyperledger Besu node
    SOVEREIGN_LEDGER_BLOCKCHAIN_RPC_URL: str = "http://localhost:8545"
    # Unlocked PoA account that pays gas (no private key needed for PoA clique)
    SOVEREIGN_LEDGER_BLOCKCHAIN_FROM_ADDRESS: str = "0x0000000000000000000000000000000000000000"
    SOVEREIGN_LEDGER_BLOCKCHAIN_CHAIN_ID: int = 1337  # Besu dev default

    # ── BASCG Provider Mode — top-level local ↔ production toggle ───────────
    # "local"      → all pillars use mock/dev providers; NAEL auto-provisioned;
    #                TSA = mock; TEE = mock; media detector = mock
    # "production" → all pillars enforce real providers; hard NAEL blocking;
    #                TSA = rfc3161 or blockchain; TEE = aws-nitro; media = api
    # Individual pillar overrides (SOVEREIGN_LEDGER_MODE, TEE_ATTESTATION_MODE,
    # etc.) always take precedence over BASCG_PROVIDER_MODE when explicitly set.
    BASCG_PROVIDER_MODE: str = "local"

    # When True and ENVIRONMENT=development, auto-issue a NAEL dev license for
    # any model that has no existing license, instead of blocking.
    NAEL_AUTO_PROVISION_DEV: bool = True

    # ── BASCG T2-A: Registry Federation ─────────────────────────────────────
    # URL of the national BASCG node to push signed registry entries to.
    # Empty string → federation disabled / local-only mode.
    # Example production value: "https://bascg.meity.gov.in"
    BASCG_NATIONAL_NODE_URL: str = ""
    # Request timeout for federation sync HTTP calls (seconds)
    BASCG_FEDERATION_TIMEOUT_SECONDS: int = 10

    # ── BASCG T3-C: Distributed TEE Attestation ──────────────────────────────
    # Enable distributed peer-challenge API and background worker
    TEE_DISTRIBUTED_ENABLED: bool = False
    # Comma-separated list of peer node base URLs to challenge periodically
    # e.g. "https://node-b.example.in,https://node-c.example.in"
    TEE_PEER_NODES: str = ""
    # Per-request timeout for challenge/respond HTTP calls (seconds)
    TEE_DISTRIBUTED_CHALLENGE_TIMEOUT_SECONDS: int = 10
    # How often to re-challenge all configured peers (minutes)
    TEE_AUTO_CHALLENGE_INTERVAL_MINUTES: int = 60

    # ── BASCG T3-B: Multi-node Policy Consensus ──────────────────────────────
    # Enable the consensus API and vote processing
    CONSENSUS_ENABLED: bool = False
    # Identifier for this node in the consensus ring (e.g. "node-hdfc-mumbai-01")
    CONSENSUS_NODE_ID: str = "local-node"
    # Fraction of received votes that must be "accept" to pass (default 2/3)
    CONSENSUS_QUORUM_THRESHOLD: float = 0.67
    # Minimum number of votes before a tally is valid
    CONSENSUS_MIN_VOTES: int = 2
    # Hours before an un-resolved proposal is automatically expired
    CONSENSUS_PROPOSAL_TTL_HOURS: int = 72
    # When True, vote signatures are verified against trusted Ed25519 keys
    CONSENSUS_VERIFY_VOTE_SIGNATURES: bool = True

    # ── DPDP AI Moderator — partner / SDK integration ────────────────────────
    # Comma-separated list of valid API keys for external partner access.
    # Each key grants access to /dpdp-moderator/* endpoints.
    # Generate per partner: python -c "import secrets; print(secrets.token_urlsafe(32))"
    # Format: "key1,key2,key3"  — whitespace around commas is stripped.
    # Empty string = API key auth disabled (JWT auth still applies to all endpoints).
    DPDP_API_KEYS: str = ""
    # Per-key rate limit: max requests per minute (sliding window).
    # Partners exceeding this receive HTTP 429. Set 0 to disable.
    DPDP_RATE_LIMIT_RPM: int = 120
    # Max items accepted in a single batch moderation request.
    DPDP_BATCH_MAX_SIZE: int = 50
    # When True, every moderation decision is appended to decisions.jsonl
    # alongside feedback.jsonl in model_output/.
    DPDP_AUDIT_LOG_ENABLED: bool = True
    # Timeout (seconds) for delivering an async moderation result to a webhook URL.
    DPDP_WEBHOOK_TIMEOUT_SECONDS: int = 30

    # ── DPDP AI Moderator — model paths ──────────────────────────────────────
    # Override the default model_output directory (absolute path or relative to cwd).
    # Leave empty to use the default path alongside inference.py.
    DPDP_MODEL_DIR: str = ""

    # ── BASCG T3-D: Legal Bundle Export ──────────────────────────────────────
    # Enable the legal bundle export API
    LEGAL_EXPORT_ENABLED: bool = True
    # Include raw TEE attestation documents (base64) in bundles — can be large
    LEGAL_EXPORT_INCLUDE_RAW_DOCUMENTS: bool = False
    # Hard cap on audit logs per bundle to prevent runaway memory usage
    LEGAL_EXPORT_MAX_AUDIT_LOGS: int = 1000
    # Ed25519-sign every exported bundle for non-repudiation
    LEGAL_EXPORT_SIGN_BUNDLES: bool = True

    # ── BASCG T3-A: Bidirectional NAIR-I Sync ────────────────────────────────
    # Enable periodic background pull from the national node
    NAIR_SYNC_ENABLED: bool = False
    # How often to run the bidirectional sync cycle (minutes)
    NAIR_SYNC_INTERVAL_MINUTES: int = 30
    # Number of entries to request per GET page when pulling
    NAIR_PULL_PAGE_SIZE: int = 100
    # When True, verify Ed25519 signature on every pulled entry
    NAIR_PULL_VERIFY_SIGNATURES: bool = True

    def get_dpdp_api_keys(self) -> set[str]:
        """Return the set of valid DPDP partner API keys."""
        if not self.DPDP_API_KEYS:
            return set()
        return {k.strip() for k in self.DPDP_API_KEYS.split(",") if k.strip()}

    def get_cors_origins(self) -> List[str]:
        """
        Returns the list of allowed CORS origins.

        IMPORTANT: `allow_credentials=True` in CORSMiddleware requires EXPLICIT
        origins — a wildcard ("*") causes browsers to silently drop every
        authenticated request.  We never return ["*"] when credentials are in use.
        """
        origins = [s.strip() for s in self.CORS_ORIGINS.split(",") if s.strip()]
        # Remove any wildcard entries — they break credentialed requests
        safe = [o for o in origins if o != "*"]
        if not safe:
            # Last-resort fallback: common Vite dev ports
            safe = [
                "http://localhost:5173", "http://127.0.0.1:5173",
                "http://localhost:5174", "http://127.0.0.1:5174",
                "http://localhost:3000", "http://127.0.0.1:3000",
            ]
        return safe

    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Allow extra env vars without crashing
    }


settings = Settings()

import json
import os

def load_thresholds():
    try:
        if os.path.exists("kavachx_thresholds.json"):
            with open("kavachx_thresholds.json", "r") as f:
                t = json.load(f)
                settings.RISK_SCORE_HIGH_THRESHOLD = t.get("risk_high", settings.RISK_SCORE_HIGH_THRESHOLD)
                settings.RISK_SCORE_MEDIUM_THRESHOLD = t.get("risk_medium", settings.RISK_SCORE_MEDIUM_THRESHOLD)
                settings.FAIRNESS_DISPARITY_THRESHOLD = t.get("fairness_disparity", settings.FAIRNESS_DISPARITY_THRESHOLD)
                settings.CONFIDENCE_LOW_THRESHOLD = t.get("confidence_low", settings.CONFIDENCE_LOW_THRESHOLD)
    except Exception as e:
        print(f"Failed to load user thresholds: {e}")

def save_thresholds(t: dict):
    with open("kavachx_thresholds.json", "w") as f:
        json.dump(t, f)
    load_thresholds()

load_thresholds()


# ── BASCG T2-B: Regulator key sidecar ───────────────────────────────────────
# Persists runtime-imported regulator public keys to bascg_regulator_keys.json
# so they survive restarts without requiring a full env-var redeploy.
# Format: {"MeitY-BASCG-v1": "<base64_pub>", "RBI-AI-Gov": "<base64_pub>"}

_REGULATOR_KEYS_FILE = "bascg_regulator_keys.json"


def load_regulator_keys() -> None:
    """
    Load persisted regulator keys from the sidecar file and merge them into
    settings.BASCG_TRUSTED_PUBLIC_KEYS_JSON.  Called at startup before
    crypto_service.initialize() so the verifier picks them up.
    """
    try:
        if os.path.exists(_REGULATOR_KEYS_FILE):
            with open(_REGULATOR_KEYS_FILE, "r") as f:
                sidecar: dict = json.load(f)
            if not isinstance(sidecar, dict) or not sidecar:
                return
            # Merge sidecar into the env-var value (env-var wins on conflict)
            existing_json = getattr(settings, "BASCG_TRUSTED_PUBLIC_KEYS_JSON", "{}").strip()
            try:
                existing: dict = json.loads(existing_json) if existing_json else {}
            except Exception:
                existing = {}
            merged = {**sidecar, **existing}   # env-var keys take precedence
            settings.BASCG_TRUSTED_PUBLIC_KEYS_JSON = json.dumps(merged)
            print(
                f"[BASCG] Loaded {len(sidecar)} regulator key(s) "
                f"from {_REGULATOR_KEYS_FILE}"
            )
    except Exception as e:
        print(f"Failed to load regulator keys sidecar: {e}")


def save_regulator_keys(keys: dict) -> None:
    """
    Persist regulator keys to the sidecar file.
    *keys* is a dict of {issuer: base64_pub_key} for ALL currently trusted
    non-local issuers (i.e., excluding dev-local).
    """
    with open(_REGULATOR_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)
    load_regulator_keys()


load_regulator_keys()
