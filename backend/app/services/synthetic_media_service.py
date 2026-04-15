"""
BASCG Phase 3 — Synthetic Media Shield Service
===============================================

Detects AI-generated / deepfake content and packages tamper-evident
evidence bundles for legal admissibility.

Architecture (Industry Standard flow — swap detector with env var only):

  MediaDetector  ─┬─ MockMediaDetector     (SYNTHETIC_MEDIA_MODE=mock,  no network)
                  ├─ APIMediaDetector      (SYNTHETIC_MEDIA_MODE=api,   HTTP call)
                  ├─ FallbackChainDetector (sequential API tier fallback)
                  └─ ConsensusDetector     (parallel API + local; math beats probability)
        │
        ▼  detection result
  SyntheticMediaShieldService.scan()
        │
        ├─ enforce policy (PASS / ALERT / BLOCK / ESCALATE)
        ├─ package evidence bundle (SHA-256 signed JSON)
        ├─ persist SyntheticMediaScanRecord to DB
        └─ Election Protection Mode → ECIWebhookService.escalate()

Legal basis:
  DPDP 2023 S.4     — Lawful processing of biometric / personal data
  IT Act S.66E      — Privacy violation by publishing private images
  IT Act S.67A/B    — Obscene electronic material
  ECI Model Code    — Synthetic political media prohibition

Configuration:
  SYNTHETIC_MEDIA_MODE            str   "mock" | "api" | "onnx"  default "mock"
  SYNTHETIC_MEDIA_API_URL         str   detector endpoint (api mode)
  SYNTHETIC_MEDIA_API_KEY         str   API key (api mode)
  SYNTHETIC_MEDIA_ONNX_MODEL_PATH str   path to .onnx model file (onnx mode)
  SYNTHETIC_MEDIA_ONNX_INPUT_NAME str   ONNX input node name  default "input"
  SYNTHETIC_MEDIA_ONNX_INPUT_SIZE int   image resize target   default 224
  SYNTHETIC_MEDIA_CONFIDENCE_THRESHOLD float  default 0.65
  ELECTION_PROTECTION_ENABLED     bool  default False
  ELECTION_PROTECTION_STATE       str   e.g. "MH" (Maharashtra)
  ECI_WEBHOOK_MODE                str   "stub" (default) | "http"
  ECI_WEBHOOK_URL                 str   ECI endpoint (http mode)
  ECI_WEBHOOK_API_KEY             str   Bearer token (http mode)
  ECI_WEBHOOK_TIMEOUT_SECONDS     int   default 10
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
import yt_dlp
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger("bascg.synthetic_media")


# ══════════════════════════════════════════════════════════════════════════════
#  §1  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """Normalised result from any media detector — provider-agnostic."""
    detector:          str
    is_synthetic:      bool
    confidence:        float           # 0.0–1.0; higher = more likely synthetic
    labels:            list            # ["GAN_face", "audio_clone", …]
    raw_response:      Optional[dict]  # full provider response (stored for audit)
    verdict:           Optional[str] = None  # human-readable verdict / fallback reason

@dataclass
class ScanResult:
    """Full result of a shield scan, ready to be persisted and returned."""
    scan_id:            str
    content_hash:       str
    content_type:       Optional[str]
    file_size_bytes:    Optional[int]
    filename:           Optional[str]
    detection:          DetectionResult
    enforcement_action: str            # PASS | ALERT | BLOCK | ESCALATE
    policy_violations:  list
    election_context:   bool
    election_state:     Optional[str]
    escalated_to_eci:   bool
    evidence_hash:      str
    evidence_bundle:    dict
    submitted_by:       Optional[str]
    source_ip:          Optional[str]
    created_at:         str


# ══════════════════════════════════════════════════════════════════════════════
#  §2  DETECTOR BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class MediaDetector:
    """Abstract base — subclasses implement detect()."""

    async def detect(self, content: bytes, content_type: Optional[str] = None) -> DetectionResult:
        raise NotImplementedError


class MockMediaDetector(MediaDetector):
    """
    Multi-signal heuristic detector for AI-generated / deepfake content.

    Detection pipeline (4 stages)
    ──────────────────────────────
    Stage 0 — Byte-level provenance (no PIL, all content types):
        • PNG tEXt/iTXt chunk scan: AUTOMATIC1111, ComfyUI, InvokeAI, NovelAI, etc.
          embed generation parameters in PNG text chunks — strongest signal for
          AI-generated logos/illustrations.  confidence = 0.92 when found.
        • Raw byte scan: "Negative prompt:", "ComfyUI", etc. in any format.

    Stage 1 — Structural analysis (PIL):
        • Image dimension analysis: AI models output at specific grid-aligned sizes
          (SDXL, SD base, DALL-E, Midjourney, etc.).  Applied BEFORE the graphic
          guard so AI logos at known AI dimensions ARE flagged.
        • PNG/WebP at AI dimensions without embedded metadata: combined origin signal.

    Stage 2a — Graphic / logo path:
        Dimension + metadata signals only.  No face-specific labels.

    Stage 2b — Photograph path:
        • EXIF Software tag (explicit AI tool → 0.90 confidence).
        • Missing camera Make/Model (real photos almost always have this).
        • Photographic PNG: cameras output JPEG, not PNG.
        • Block-level noise floor: real sensor photos have shot noise (std ≥ 3);
          AI diffusion images can have near-zero noise in uniform regions.
        • Skin-tone proxy: gates face-specific labels.
        • Edge-smoothness proxy: AI upscaling produces unnaturally smooth gradients.

    is_synthetic threshold: 0.40 (shows "suspicious" earlier on frontend).
    Enforcement action ALERT still uses the service-level threshold (default 0.65).

    Labels
        ai_generation_metadata  explicit AI tool marker in file metadata
        ai_generated            AI-generated logo/graphic (no face content)
        diffusion_upscale       photo-level diffusion / upscale detected
        GAN_face                face-like content + AI signal

    Test vectors
        b"SYNTHETIC_TEST" → confidence 0.95, labels ["GAN_face","diffusion_upscale"]
        b"REAL_TEST"      → confidence 0.03, labels []
    """

    _ENFORCEMENT_THRESHOLD = 0.45  # Align with governance default
    _KNOWN_AI_HASHES: dict = {}    # Session-level perceptual hash → confidence cache (bounded to 1000)

    _AI_TOOL_KEYWORDS: list = [
        "midjourney", "stable diffusion", "dall-e", "dall·e",
        "firefly", "generative fill", "openai", "runway", "pika",
        "kling", "sora", "dreamstudio", "adobe generative",
        "comfyui", "automatic1111", "novelai", "imagine",
        "invokeai", "leonardo", "adobe firefly",
    ]

    async def detect(
        self, content: bytes, content_type: Optional[str] = None
    ) -> DetectionResult:
        # ── Test vectors ──────────────────────────────────────────────────────
        if content == b"SYNTHETIC_TEST":
            return DetectionResult(
                detector="mock-heuristic", is_synthetic=True, confidence=0.95,
                labels=["GAN_face", "diffusion_upscale"],
                raw_response={"method": "test_vector", "vector": "SYNTHETIC_TEST"},
            )
        if content == b"REAL_TEST":
            return DetectionResult(
                detector="mock-heuristic", is_synthetic=False, confidence=0.03,
                labels=[], raw_response={"method": "test_vector", "vector": "REAL_TEST"},
            )

        ct = (content_type or "").lower()

        if ct.startswith("image/"):
            return self._analyze_image(content, ct)
        if ct.startswith("audio/"):
            return self._analyze_audio(content, ct)
        if ct.startswith("video/"):
            return self._analyze_video(content, ct)

        # Unknown / unsupported content type — conservative pass
        return DetectionResult(
            detector="mock-heuristic", is_synthetic=False, confidence=0.02,
            labels=[],
            raw_response={"method": "no_handler", "content_type": content_type},
        )

    # ── Stage-0 helpers (no PIL needed) ───────────────────────────────────────

    @staticmethod
    def _scan_png_chunks(content: bytes) -> dict:
        """
        Parse PNG tEXt/iTXt chunks for AI generation metadata.

        Known embedders:
          AUTOMATIC1111 SD  → key "parameters", value starts "Steps: … Sampler: …"
          ComfyUI           → key "prompt" (JSON) or "workflow"
          InvokeAI          → key "invokeai_metadata" or "sd-metadata"
          NovelAI           → key "Comment" with AI params
          Any SD fork       → value contains "Negative prompt:"
        """
        PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
        if not content.startswith(PNG_MAGIC):
            return {"found": False}

        # Keys written by known AI generation tools
        _AI_PNG_KEYS = {
            "parameters", "prompt", "workflow", "sd-metadata",
            "invokeai_metadata", "source", "comment", "description",
        }
        # Substrings that appear in AI generation parameter blocks
        _AI_VALUE_KEYWORDS = [
            "negative prompt:", "steps:", "sampler:", "cfg scale:",
            "model_name", "sampler_name",  # ComfyUI JSON
            "comfyui", "invokeai", "novelai", "midjourney",
            "stable diffusion", "dall-e", "openai", "automatic1111",
        ]

        pos = 8  # skip PNG signature
        while pos + 12 <= len(content):
            try:
                length     = int.from_bytes(content[pos:pos+4], 'big')
                chunk_type = content[pos+4:pos+8]
                chunk_data = content[pos+8:pos+8+length]
                pos       += 12 + length
            except Exception:
                break

            if chunk_type in (b'tEXt', b'iTXt'):
                try:
                    null_idx = chunk_data.index(b'\x00')
                    key      = chunk_data[:null_idx].decode('latin-1', errors='replace').strip().lower()
                    val      = chunk_data[null_idx+1:].decode('latin-1', errors='replace').lower()

                    if key in _AI_PNG_KEYS:
                        return {"found": True, "key": key, "keyword": f"key:{key}"}

                    for kw in _AI_VALUE_KEYWORDS:
                        if kw in val:
                            return {"found": True, "key": key, "keyword": kw}
                except (ValueError, Exception):
                    pass

            if chunk_type == b'IEND':
                break

        return {"found": False}

    @staticmethod
    def _scan_raw_bytes(content: bytes) -> dict:
        """
        Scan raw file bytes for AI tool signatures.
        Checks first 16 KB (EXIF, XMP, IPTC) and last 2 KB (some tools write at end).
        """
        # Byte signatures embedded by AI generation tools
        _AI_BYTE_SIGS = [
            b"Negative prompt:",       # AUTOMATIC1111 (nearly universal in SD outputs)
            b"Steps: ",                # AUTOMATIC1111
            b"Sampler: ",              # AUTOMATIC1111
            b"CFG scale: ",            # AUTOMATIC1111
            b"ComfyUI",                # ComfyUI
            b"invokeai",               # InvokeAI
            b"NovelAI",                # NovelAI
            b'"model_name"',           # ComfyUI JSON
            b'"sampler_name"',         # ComfyUI JSON
            b"sd-metadata",            # InvokeAI / various
            b"ai-generated",           # generic AI watermark
        ]

        scan_zone = content[:16384]
        if len(content) > 16384:
            scan_zone += content[-2048:]

        for sig in _AI_BYTE_SIGS:
            if sig in scan_zone:
                return {"found": True, "signature": sig.decode('latin-1', errors='replace').strip()}

        return {"found": False}

    # ── Stage-1 structural helpers (no network, no ML) ────────────────────────

    @staticmethod
    def _check_dimensions(w: int, h: int) -> tuple:
        """
        Returns (signal_weight, description) for AI-typical image dimensions.
        Returns (0.0, "") when dimensions don't match any known AI output size.

        Scoring rationale:
          SDXL_EXACT  (0.32) — extremely uncommon in real photos or human-made assets
          AI_STD      (0.20) — standard output of DALL-E, Midjourney v5-v7, SD 2.x
          AI_BASE     (0.10) — overlap with real-world icons; weak signal only
          Power-of-2  (0.18) — DALL-E logo output format, SD hires fix
          64-grid     (0.10) — latent grid alignment; weak corroborative signal
        """
        # SDXL-native: almost never appear in real photos or human-designed assets
        _SDXL_EXACT = {
            (1152, 896), (896, 1152), (1216, 832), (832, 1216),
            (1344, 768), (768, 1344), (1536, 640), (640, 1536),
            (1280, 768), (768, 1280),
        }
        # Strong AI output sizes (DALL-E 3, Midjourney v5-v7, SD 2.x, FLUX)
        _AI_STD = {
            (1024, 1024), (1024, 1792), (1792, 1024),
            (1456, 816),  (816, 1456),  (1360, 768),  (768, 1360),
            (1280, 1280), (1024, 576),  (576, 1024),
        }
        # Common AI base sizes (also used for real icons/logos — weak signal)
        _AI_BASE = {
            (512, 512),  (768, 768),  (512, 768),  (768, 512),
            (512, 640),  (640, 512),  (512, 896),  (896, 512),
            (512, 1024), (1024, 512), (640, 1024), (1024, 640),
        }

        if (w, h) in _SDXL_EXACT:
            return 0.32, f"SDXL standard output ({w}×{h})"
        if (w, h) in _AI_STD:
            return 0.20, f"AI generation standard size ({w}×{h})"
        if (w, h) in _AI_BASE:
            return 0.10, f"Common AI base model size ({w}×{h})"
        # Power-of-2 square ≥ 1024 (DALL-E logo outputs, SD hires)
        if w == h and w >= 1024 and (w & (w - 1)) == 0:
            return 0.18, f"Power-of-2 square ≥ 1024 ({w}×{h})"
        # 64-pixel latent grid, large enough to be meaningful
        if 512 <= w <= 2048 and 512 <= h <= 2048 and w % 64 == 0 and h % 64 == 0:
            return 0.10, f"64-pixel AI latent grid ({w}×{h})"
        return 0.0, ""

    @staticmethod
    def _analyze_noise_floor(r_vals: list) -> tuple:
        """
        Block-level noise analysis on the R channel of a 64×64 thumbnail.

        Real digital cameras produce shot noise — even in the smoothest region
        of a real photo the local standard deviation is typically ≥ 3.0.
        AI diffusion images can have near-zero noise in flat areas: a tell-tale
        "too clean" artefact.

        Returns (signal_weight, min_block_std).
        """
        if len(r_vals) < 4096:   # need a 64×64 thumbnail
            return 0.0, 10.0

        block_stds = []
        for by in range(4):
            for bx in range(4):
                block = []
                for py in range(16):
                    s = (by * 16 + py) * 64 + bx * 16
                    block.extend(r_vals[s: s + 16])
                if block:
                    bm   = sum(block) / len(block)
                    bstd = (sum((x - bm) ** 2 for x in block) / len(block)) ** 0.5
                    block_stds.append(bstd)

        if not block_stds:
            return 0.0, 10.0

        min_std    = min(block_stds)
        low_blocks = sum(1 for s in block_stds if s < 3.0)

        if min_std < 0.8:
            return 0.22, min_std            # near-zero noise — very suspicious
        if min_std < 2.0:
            return 0.16, min_std            # low noise floor — suspicious
        if min_std < 3.5 and low_blocks >= 6:
            return 0.10, min_std            # several abnormally smooth blocks
        return 0.0, min_std

    # ── New forensic analysis helpers ─────────────────────────────────────────

    @staticmethod
    def _check_c2pa(content: bytes) -> dict:
        """
        Detect C2PA / CAI provenance markers.  Presence means a camera or trusted
        editor cryptographically signed this file's origin — strong authenticity
        indicator.  Absence is neutral (not all real images carry C2PA yet).

        Checks: JUMBF box header ('jumb'), C2PA namespace ('c2pa'), JPEG APP11
        marker (0xFF 0xEB), Content Authenticity Bundle ('cab '), CAI namespace.
        """
        _SIGS = [b'jumb', b'c2pa', b'cab ', b'cai.', b'\xff\xeb']
        zone  = content[:32768]
        for sig in _SIGS:
            if sig in zone:
                return {"found": True, "marker": sig.decode("latin-1", errors="replace").strip()}
        return {"found": False}

    @staticmethod
    def _compute_dhash(img) -> int:
        """
        64-bit difference hash (dHash) for near-duplicate detection.
        Resize to 9×8, compare adjacent horizontal pixels → 64-bit int.
        """
        from PIL import Image as _PIL_Image
        tiny   = img.convert("L").resize((9, 8), _PIL_Image.LANCZOS)
        pixels = list(tiny.getdata())
        h = 0
        for row in range(8):
            for col in range(8):
                if pixels[row * 9 + col] > pixels[row * 9 + col + 1]:
                    h |= 1 << (row * 8 + col)
        return h

    @staticmethod
    def _ela_analysis(content: bytes, content_type: str, img) -> tuple:
        """
        Compression-ratio ELA: re-save the image as JPEG at quality=75 and compare
        sizes.  AI-generated images that have never been JPEG-compressed before
        compress to 40-60 % of their original size on the first pass.  Real photos
        that were already JPEG-encoded stay close to their original size.

        Returns (signal_weight, compression_ratio).
        """
        try:
            import io as _io
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=75, optimize=True)
            ratio = len(buf.getvalue()) / max(len(content), 1)
            if content_type == "image/jpeg":
                if ratio < 0.45:
                    return 0.18, ratio   # very high gain — highly suspicious first-compression
                if ratio < 0.62:
                    return 0.10, ratio   # moderate gain — corroborative
        except Exception:
            pass
        return 0.0, 1.0

    @staticmethod
    def _fft_grid_artifact(img) -> tuple:
        """
        Detect periodic grid artifacts from diffusion model VAE upsampling.
        Diffusion models run at 1/8 (SD) or 1/16 (SDXL) latent scale and
        upsample back to pixel space, leaving spectral peaks at multiples of
        the latent tile size.

        Requires numpy; silently returns (0.0, '') if not available.
        Returns (signal_weight, description).
        """
        try:
            import numpy as np
            gray   = np.array(img.convert("L").resize((256, 256)), dtype=np.float32)
            fshift = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
            cy, cx = 128, 128
            fshift[cy - 4:cy + 4, cx - 4:cx + 4] = 0  # suppress DC
            mean_e = fshift.mean() or 1e-9
            # Spectral indices for latent grid periods in a 256-px image:
            #   8×  downsampling → period 32 px → spectral index 8
            #   16× downsampling → period 16 px → spectral index 16
            for sidx, label in [(8, "32px"), (16, "16px"), (32, "8px")]:
                band = (
                    fshift[cy - sidx - 2:cy - sidx + 2, :].mean()
                    + fshift[cy + sidx - 2:cy + sidx + 2, :].mean()
                    + fshift[:, cx - sidx - 2:cx - sidx + 2].mean()
                    + fshift[:, cx + sidx - 2:cx + sidx + 2].mean()
                ) / 4
                ratio = band / mean_e
                if ratio > 6.0:
                    return 0.18, f"Spectral grid at {label} period (VAE upsampling artifact)"
                if ratio > 4.0:
                    return 0.10, f"Weak spectral periodicity at {label} period"
        except Exception:
            pass
        return 0.0, ""

    # ── Image analysis ────────────────────────────────────────────────────────

    def _analyze_image(self, content: bytes, content_type: str) -> DetectionResult:
        """
        4-stage image analysis.

        Critical design decision: Stage 1 dimension analysis runs BEFORE the
        graphic/logo early-return so AI logos at known AI dimensions ARE flagged
        even when they carry no embedded metadata (DALL-E, Midjourney web exports).
        """
        signals:  dict = {}
        labels:   list = []
        raw_data: dict = {}

        # ── Stage 0: File-level metadata (all image types) ────────────────────
        png_meta = self._scan_png_chunks(content)
        if png_meta["found"]:
            signals["ai_png_metadata"] = 0.92
            labels.append("ai_generation_metadata")
            raw_data.update({"png_ai_key": png_meta.get("key"), "png_ai_keyword": png_meta.get("keyword")})
        else:
            byte_scan = self._scan_raw_bytes(content)
            if byte_scan["found"]:
                signals["ai_byte_signature"] = 0.85
                labels.append("ai_generation_metadata")
                raw_data["byte_signature"] = byte_scan.get("signature")

        # C2PA / Content Authenticity Initiative provenance (all images)
        c2pa = self._check_c2pa(content)
        raw_data["c2pa_provenance"] = c2pa["found"]
        if c2pa["found"]:
            raw_data["c2pa_marker"] = c2pa.get("marker")

        # ── Stage 1: PIL load ─────────────────────────────────────────────────
        try:
            from PIL import Image
            import io as _io
        except ImportError:
            logger.warning("Pillow not installed — image heuristics unavailable")
            return self._finalize(signals, labels, raw_data, "pil_unavailable", False)

        try:
            img = Image.open(_io.BytesIO(content))
            img.load()
        except Exception as exc:
            logger.warning("Image decode failed: %s", exc)
            return self._finalize(signals, labels, raw_data, "decode_error", False)

        w, h = img.size
        raw_data["dimensions"] = f"{w}×{h}"

        # Perceptual hash — check against session-level known-AI cache
        try:
            dhash = self._compute_dhash(img)
            raw_data["dhash"] = dhash
            for known_h, known_conf in list(self._KNOWN_AI_HASHES.items())[:500]:
                if bin(dhash ^ known_h).count("1") <= 8:  # Hamming ≤ 8 = near-duplicate
                    signals["known_ai_hash_match"] = min(0.85, known_conf + 0.10)
                    raw_data["hash_match_conf"] = round(known_conf, 4)
                    break
        except Exception:
            pass

        # ── Stage 1: Dimension analysis (ALL images, before graphic guard) ─────
        dim_signal, dim_label = self._check_dimensions(w, h)
        if dim_signal > 0:
            signals["ai_typical_dimensions"] = dim_signal
            raw_data["dimension_reason"] = dim_label

        # Combined signal: PNG/WebP at AI dimensions with zero embedded metadata.
        # Real brand assets rarely have AI-standard dimensions AND no origin metadata.
        is_png = content_type in ("image/png", "image/webp")
        has_meta_signal = bool(signals.get("ai_png_metadata") or signals.get("ai_byte_signature"))
        if is_png and dim_signal >= 0.10 and not has_meta_signal:
            signals["ai_dimensions_no_embedded_metadata"] = 0.14
            raw_data["no_embedded_metadata"] = True

        # ── Graphic / logo classification ─────────────────────────────────────
        has_alpha = img.mode in ("RGBA", "LA", "PA")
        thumb     = img.convert("RGB").resize((64, 64), Image.LANCZOS)
        pixels    = list(thumb.getdata())
        n         = len(pixels)
        r_vals    = [p[0] for p in pixels]
        g_vals    = [p[1] for p in pixels]
        b_vals    = [p[2] for p in pixels]
        mr, mg, mb = sum(r_vals)/n, sum(g_vals)/n, sum(b_vals)/n
        color_std = ((
            sum((x-mr)**2 for x in r_vals) +
            sum((x-mg)**2 for x in g_vals) +
            sum((x-mb)**2 for x in b_vals)
        ) / (3*n)) ** 0.5

        unique_colors  = len(set(pixels))
        # Refined Guard: A 'Photograph' must have high color standard deviation (variance).
        # Logically: it's a graphic ONLY if it has few colors AND low variance.
        # This prevents AI-generated photos of people from being incorrectly skipped.
        is_graphic     = (unique_colors < 150 or has_alpha) and color_std < 20.0
        classification = "graphic_or_logo" if is_graphic else "photograph"

        raw_data.update({
            "classification": classification,
            "unique_colors":  unique_colors,
            "color_std":      round(color_std, 2),
            "has_alpha":      has_alpha,
        })

        if is_graphic:
            return self._finalize(signals, labels, raw_data, classification, False)

        # ── Stage 2b: Photograph-specific signals ─────────────────────────────
        has_camera_exif = False
        ai_tool_name    = None
        try:
            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif:
                make  = str(raw_exif.get(271, "") or "").strip()
                model = str(raw_exif.get(272, "") or "").strip()
                has_camera_exif = bool(make or model)
                software = str(raw_exif.get(305, "") or "").lower()
                for kw in self._AI_TOOL_KEYWORDS:
                    if kw in software:
                        if "ai_generation_metadata" not in labels:
                            labels.append("ai_generation_metadata")
                        signals["ai_exif_software"] = 0.90
                        ai_tool_name = kw
                        raw_data["ai_tool_in_exif"] = kw
                        break
        except Exception:
            pass

        # Missing camera EXIF — real photos almost always carry Make/Model
        if not has_camera_exif and "ai_exif_software" not in signals:
            if content_type == "image/jpeg":
                signals["missing_camera_exif"] = 0.22   # strong signal for JPEG
            elif is_png:
                signals["missing_camera_exif"] = 0.10   # weaker for PNG (screenshots etc.)

        # Photographic PNG: cameras produce JPEG, not PNG
        if is_png and classification == "photograph":
            signals["photographic_png_no_camera_origin"] = 0.14
            raw_data["note"] = "Photographic PNG — real cameras output JPEG"

        # Block-level noise floor analysis
        noise_signal, noise_floor = self._analyze_noise_floor(r_vals)
        if noise_signal > 0:
            signals["anomalous_noise_floor"] = noise_signal
            raw_data["min_block_noise"] = round(noise_floor, 2)

        # Skin-tone proxy (gates face-specific labels)
        skin_count = sum(
            1 for r, g, b in pixels
            if r > 95 and g > 40 and b > 20
            and r > g and r > b
            and abs(r - g) > 15
            and (r - b) > 15
        )
        skin_ratio       = skin_count / n
        has_face_content = skin_ratio > 0.07

        # Edge smoothness (AI diffusion → unnaturally smooth gradients)
        diffs     = [abs(r_vals[i]-r_vals[i-1]) for i in range(1, n) if i % 64 != 0]
        mean_edge = sum(diffs) / max(len(diffs), 1)
        if mean_edge < 6.0:
            signals["smooth_texture"] = 0.12

        raw_data.update({
            "has_camera_exif":  has_camera_exif,
            "ai_tool_detected": ai_tool_name,
            "skin_ratio":       round(skin_ratio, 3),
            "has_face_content": has_face_content,
            "mean_edge_diff":   round(mean_edge, 3),
        })

        # ── Stage 3: Compression-ratio ELA ───────────────────────────────────
        ela_signal, ela_ratio = self._ela_analysis(content, content_type, img)
        if ela_signal > 0:
            signals["ela_compression_ratio"] = ela_signal
            raw_data["ela_ratio"] = round(ela_ratio, 3)

        # ── Stage 4: FFT grid artifact detection (numpy-gated) ───────────────
        fft_signal, fft_label = self._fft_grid_artifact(img)
        if fft_signal > 0:
            signals["fft_grid_artifact"] = fft_signal
            raw_data["fft_label"] = fft_label

        return self._finalize(signals, labels, raw_data, classification, has_face_content)

    def _finalize(
        self,
        signals:          dict,
        labels:           list,
        raw_data:         dict,
        classification:   str,
        has_face_content: bool,
    ) -> DetectionResult:
        """Compute confidence, assign labels, and categorise signals for the frontend."""
        confidence   = min(0.95, sum(signals.values()))
        is_synthetic = confidence >= self._ENFORCEMENT_THRESHOLD

        # Assign content-appropriate labels
        if is_synthetic:
            if classification in ("graphic_or_logo", "pil_unavailable", "decode_error"):
                if "ai_generated" not in labels:
                    labels.append("ai_generated")
            elif has_face_content:
                if "GAN_face" not in labels:
                    labels.extend(["GAN_face", "diffusion_upscale"])
            else:
                if "diffusion_upscale" not in labels:
                    labels.append("diffusion_upscale")

        # Group signals by detection category for frontend display
        _CAT = {
            # Metadata / provenance
            "ai_png_metadata":                      "metadata",
            "ai_byte_signature":                    "metadata",
            "ai_exif_software":                     "metadata",
            "known_ai_hash_match":                  "metadata",
            # Structural / origin
            "ai_typical_dimensions":                "structural",
            "ai_dimensions_no_embedded_metadata":   "structural",
            "missing_camera_exif":                  "structural",
            "photographic_png_no_camera_origin":    "structural",
            # Structural (video / audio)
            "ffmpeg_encoder":                       "structural",
            "deepfake_typical_resolution":          "structural",
            "video_frame_ai_signals":               "pixel_analysis",
            # Pixel-level forensics
            "anomalous_noise_floor":                "pixel_analysis",
            "smooth_texture":                       "pixel_analysis",
            "ela_compression_ratio":                "pixel_analysis",
            "fft_grid_artifact":                    "pixel_analysis",
            # Audio spectral
            "tts_abnormal_zcr":                     "other",
            "tts_uniform_amplitude":                "other",
            "tts_flat_energy":                      "other",
        }
        categories: dict = {}
        for sig, score in signals.items():
            cat = _CAT.get(sig, "other")
            categories[cat] = round(min(0.95, categories.get(cat, 0) + score), 4)

        # Register dhash of confirmed synthetic images for future re-upload detection
        if is_synthetic and "dhash" in raw_data:
            self._KNOWN_AI_HASHES[raw_data["dhash"]] = confidence
            if len(self._KNOWN_AI_HASHES) > 1000:   # keep cache bounded
                try:
                    del self._KNOWN_AI_HASHES[next(iter(self._KNOWN_AI_HASHES))]
                except StopIteration:
                    pass

        return DetectionResult(
            detector="mock-heuristic",
            is_synthetic=is_synthetic,
            confidence=round(confidence, 4),
            labels=labels,
            raw_response={
                "method":            "heuristic_image",
                "signals":           {k: round(v, 4) for k, v in signals.items()},
                "signal_categories": categories,
                **raw_data,
            },
        )

    # ── Audio analysis ────────────────────────────────────────────────────────

    def _analyze_audio(self, content: bytes, content_type: str) -> DetectionResult:
        """
        File-level audio heuristics: WAV RIFF header analysis, MP3 frame inspection,
        silence-pattern detection, and mono/sample-rate fingerprinting common in TTS.
        """
        signals: dict = {}
        labels:  list = []
        raw_data: dict = {"content_type": content_type, "size_kb": round(len(content) / 1024, 1)}

        if len(content) < 1024 * 10:
            signals["very_short_audio"] = 0.10

        # ── WAV (RIFF) header analysis ────────────────────────────────────────
        if content_type in ("audio/wav", "audio/x-wav") and len(content) > 44:
            try:
                if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
                    channels = int.from_bytes(content[22:24], "little")
                    sr       = int.from_bytes(content[24:28], "little")
                    bits     = int.from_bytes(content[34:36], "little")
                    raw_data.update({"wav_channels": channels, "wav_sr": sr, "wav_bits": bits})

                    # TTS-typical sample rates: 8k, 16k, 22.05k, 24k Hz
                    if sr in (8000, 16000, 22050, 24000):
                        signals["tts_sample_rate"] = 0.15

                    # Mono 16-bit PCM at ≥16 kHz: canonical TTS output format
                    if channels == 1 and sr >= 16000 and bits == 16:
                        signals["tts_mono_speech"] = 0.10

                    # Silence padding: TTS engines pad output at boundaries
                    audio_data = content[44: 44 + min(len(content) - 44, 8192)]
                    if len(audio_data) >= 400:
                        near_zero = sum(
                            1 for i in range(0, len(audio_data) - 1, 2)
                            if abs(int.from_bytes(audio_data[i:i + 2], "little", signed=True)) < 64
                        )
                        silence_ratio = near_zero / max(len(audio_data) // 2, 1)
                        raw_data["silence_ratio"] = round(silence_ratio, 3)
                        if silence_ratio > 0.35:
                            signals["tts_silence_padding"] = 0.12

                    # ── Sample-level spectral proxy (ZCR, amplitude CV, energy CV) ──
                    samp_signals, samp_metrics = MockMediaDetector._analyze_audio_samples(content, sr)
                    signals.update(samp_signals)
                    raw_data.update(samp_metrics)
            except Exception:
                pass

        # ── MP3 frame header inspection ───────────────────────────────────────
        if content_type == "audio/mpeg" and len(content) > 20:
            try:
                for i in range(min(1000, len(content) - 3)):
                    if content[i] == 0xFF and (content[i + 1] & 0xE0) == 0xE0:
                        br_idx  = (content[i + 2] >> 4) & 0x0F
                        br_tbl  = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
                        bitrate = br_tbl[br_idx] if 1 <= br_idx <= 14 else 0
                        has_id3 = content[:3] == b"ID3"
                        raw_data.update({"mp3_bitrate": bitrate, "has_id3": has_id3})
                        # TTS output at 64/128 kbps without ID3 metadata = common synthesis
                        if bitrate in (64, 128) and not has_id3:
                            signals["tts_cbr_mp3"] = 0.12
                        break
            except Exception:
                pass

        confidence   = min(0.95, sum(signals.values()))
        is_synthetic = confidence >= self._ENFORCEMENT_THRESHOLD
        if is_synthetic:
            labels.append("audio_clone")

        return DetectionResult(
            detector="mock-heuristic",
            is_synthetic=is_synthetic,
            confidence=round(confidence, 4),
            labels=labels,
            raw_response={
                "method":           "heuristic_audio",
                "signals":          {k: round(v, 4) for k, v in signals.items()},
                "signal_categories": {"structural": round(min(0.95, sum(signals.values())), 4)},
                **raw_data,
                "note": "Full spectral analysis (MFCCs, pitch contour) requires librosa — header heuristics only",
            },
        )

    # ── Video analysis ────────────────────────────────────────────────────────

    def _analyze_video(self, content: bytes, content_type: str) -> DetectionResult:
        """
        Container-level video heuristics: ftyp/brand analysis, encoder fingerprints,
        and deepfake-typical square resolutions in MP4 tkhd box.
        Full frame-level detection requires ffmpeg/cv2.
        """
        signals: dict = {}
        labels:  list = []
        raw_data: dict = {"content_type": content_type, "size_mb": round(len(content) / (1024 * 1024), 2)}
        scan = content[:min(65536, len(content))]

        if len(content) < 1024 * 100:
            signals["suspiciously_small"] = 0.10

        # ── MP4 / MOV container analysis ─────────────────────────────────────
        if content_type in ("video/mp4", "video/quicktime", "video/x-m4v"):
            # ftyp box: bytes 4-8 = 'ftyp', bytes 8-12 = major brand
            if len(content) >= 12 and content[4:8] == b"ftyp":
                brand = content[8:12].decode("ascii", errors="replace").strip()
                raw_data["ftyp_brand"] = brand

            # Many deepfake pipelines (FaceSwap, DeepFaceLab, SimSwap) use ffmpeg-based encoding
            _ENC_SIGS = [b"Lavf", b"Lavc", b"libx264", b"libx265", b"FFMPEG"]
            for sig in _ENC_SIGS:
                if sig in scan:
                    signals["ffmpeg_encoder"] = 0.08
                    raw_data["encoder_hint"] = sig.decode("ascii", errors="replace")
                    break

            # Deepfake-typical square crop resolutions stored as 16.16 fixed-point in tkhd box
            # 256×256 → \x00\x01\x00\x00 \x00\x01\x00\x00
            # 512×512 → \x00\x02\x00\x00 \x00\x02\x00\x00
            _DEEPFAKE_DIMS = [
                b"\x00\x01\x00\x00\x00\x01\x00\x00",  # 256×256
                b"\x00\x02\x00\x00\x00\x02\x00\x00",  # 512×512
            ]
            for dim in _DEEPFAKE_DIMS:
                if dim in scan:
                    signals["deepfake_typical_resolution"] = 0.12
                    raw_data["deepfake_dim_hint"] = True
                    break

        # ── WebM / MKV container ──────────────────────────────────────────────
        if content_type in ("video/webm", "video/x-matroska"):
            if b"V_VP9" in scan or b"V_VP8" in scan:
                raw_data["video_codec"] = "VP9/VP8"
            # ffmpeg ENCODER tag inside EBML/WebM header
            if b"ENCODER" in scan and b"ffmpeg" in scan:
                signals["ffmpeg_encoder"] = 0.08

        # ── Embedded JPEG frame extraction (MP4/MOV) ─────────────────────────
        # Pull a keyframe from the container and run the full image pipeline on it.
        if content_type in ("video/mp4", "video/quicktime", "video/x-m4v"):
            try:
                thumb = MockMediaDetector._extract_mp4_thumbnail(content)
                if thumb is not None:
                    frame_result = self._analyze_image(thumb, "image/jpeg")
                    raw_data["embedded_frame_analyzed"] = True
                    raw_data["frame_confidence"] = round(frame_result.confidence, 4)
                    raw_data["frame_labels"] = frame_result.labels
                    if frame_result.confidence > 0.15:
                        weight = min(0.28, frame_result.confidence * 0.50)
                        signals["video_frame_ai_signals"] = round(weight, 4)
                        for lbl in frame_result.labels:
                            if lbl not in labels:
                                labels.append(lbl)
            except Exception:
                pass

        confidence   = min(0.95, sum(signals.values()))
        is_synthetic = confidence >= self._ENFORCEMENT_THRESHOLD
        if is_synthetic:
            labels.append("lip_sync")

        return DetectionResult(
            detector="mock-heuristic",
            is_synthetic=is_synthetic,
            confidence=round(confidence, 4),
            labels=labels,
            raw_response={
                "method":           "heuristic_video",
                "signals":          {k: round(v, 4) for k, v in signals.items()},
                "signal_categories": {"structural": round(min(0.95, sum(signals.values())), 4)},
                **raw_data,
                "note": "Full frame analysis requires ffmpeg/cv2 — container heuristics only",
            },
        )

    # ── Audio sample-level spectral proxy ─────────────────────────────────────

    @staticmethod
    def _analyze_audio_samples(content: bytes, sample_rate: int) -> tuple:
        """
        Pure-Python PCM sample analysis for TTS / voice-clone detection.
        No external libraries required — parses 16-bit signed PCM from raw WAV bytes.

        Signals emitted:
          tts_abnormal_zcr      — ZCR outside normal speech range (0.05–0.25)
          tts_uniform_amplitude — amplitude coefficient of variation too low (over-smooth)
                                  or too high (synthesis clipping)
          tts_flat_energy       — short-term energy variance too low (monotone profile)

        Returns (signals_dict, metrics_dict).
        """
        signals: dict = {}
        metrics: dict = {}

        # Parse 16-bit signed PCM (skip 44-byte standard RIFF/PCM header)
        audio_data = content[44:]
        if len(audio_data) < 400:
            return signals, metrics

        # Analyse first 2 s worth of samples — cap at 8192 samples for speed
        max_samples = min(len(audio_data) // 2, sample_rate * 2, 8192)
        samples: list = []
        for i in range(max_samples):
            offset = i * 2
            if offset + 2 <= len(audio_data):
                s = int.from_bytes(audio_data[offset:offset + 2], "little", signed=True)
                samples.append(s)

        if len(samples) < 64:
            return signals, metrics

        # ── Zero Crossing Rate ────────────────────────────────────────────────
        # Real speech: irregular ZCR 0.05–0.25.
        # TTS: hyper-smooth pitch → very low ZCR; or clipped → very high ZCR.
        zero_crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i - 1] < 0) != (samples[i] < 0)
        )
        zcr = zero_crossings / len(samples)
        metrics["zero_crossing_rate"] = round(zcr, 4)
        if zcr < 0.04 or zcr > 0.30:
            signals["tts_abnormal_zcr"] = 0.10

        # ── Amplitude Envelope Consistency ────────────────────────────────────
        # TTS normalisation produces unnaturally uniform amplitude.
        # Coefficient of variation (std/mean) of absolute sample values:
        #   Real speech: CV ~0.6–1.5 (high dynamic range)
        #   Over-smooth TTS: CV < 0.40; clipped synthesis: CV > 2.5
        abs_samples = [abs(s) for s in samples]
        mean_amp = sum(abs_samples) / len(abs_samples) if abs_samples else 0
        if mean_amp > 0:
            variance = sum((x - mean_amp) ** 2 for x in abs_samples) / len(abs_samples)
            std_amp  = variance ** 0.5
            cv       = std_amp / mean_amp
            metrics["amplitude_cv"] = round(cv, 4)
            if cv < 0.40:
                signals["tts_uniform_amplitude"] = 0.08   # over-normalised
            elif cv > 2.5:
                signals["tts_uniform_amplitude"] = 0.16   # clipping artefact

        # ── Short-term Energy Variance ────────────────────────────────────────
        # Divide into 16-sample frames and measure frame-energy CV.
        # Real speech alternates stressed/unstressed → high energy variance.
        # TTS: flat energy profile → low CV.
        frame_size = 16
        frame_energies: list = []
        for i in range(0, len(samples) - frame_size, frame_size):
            frame  = samples[i:i + frame_size]
            energy = sum(s * s for s in frame) / frame_size
            frame_energies.append(energy)

        if len(frame_energies) >= 4:
            mean_e = sum(frame_energies) / len(frame_energies)
            if mean_e > 0:
                var_e     = sum((e - mean_e) ** 2 for e in frame_energies) / len(frame_energies)
                energy_cv = (var_e ** 0.5) / mean_e
                metrics["energy_cv"] = round(energy_cv, 4)
                if energy_cv < 0.35:
                    signals["tts_flat_energy"] = 0.12

        return signals, metrics

    # ── Video embedded frame extraction ───────────────────────────────────────

    @staticmethod
    def _extract_mp4_thumbnail(content: bytes):
        """
        Scan the first 4 MiB of an MP4/MOV container for an embedded JPEG keyframe.

        Strategy: locate JPEG SOI (\\xff\\xd8\\xff) followed by a valid JFIF/EXIF/SOF
        marker, then find the matching EOI (\\xff\\xd9).  Validate with PIL and require
        minimum 64×64 px before returning.

        Returns JPEG bytes if found, else None.
        """
        try:
            from PIL import Image
            import io
        except ImportError:
            return None

        scan_limit = min(len(content), 4 * 1024 * 1024)
        scan       = content[:scan_limit]
        pos        = 0

        while pos < len(scan) - 3:
            idx = scan.find(b"\xff\xd8\xff", pos)
            if idx == -1:
                break

            # Byte after FF D8 must be a valid JPEG marker
            next_byte = scan[idx + 2]
            valid_markers = set(range(0xE0, 0xF0)) | {0xDB, 0xC0, 0xC2, 0xFE, 0xE0}
            if next_byte not in valid_markers:
                pos = idx + 1
                continue

            # Find EOI — require minimum of 100 bytes for a real JPEG
            end_idx = scan.find(b"\xff\xd9", idx + 4)
            if end_idx == -1 or (end_idx - idx) < 100:
                pos = idx + 1
                continue

            jpeg_bytes = scan[idx:end_idx + 2]
            try:
                img = Image.open(io.BytesIO(jpeg_bytes))
                img.verify()                          # checks integrity
                img2 = Image.open(io.BytesIO(jpeg_bytes))  # re-open after verify
                w, h = img2.size
                if w >= 64 and h >= 64:
                    return jpeg_bytes
            except Exception:
                pass

            pos = idx + 1

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_pass(reason: str) -> DetectionResult:
        return DetectionResult(
            detector="mock-heuristic", is_synthetic=False, confidence=0.03,
            labels=[], raw_response={"method": "safe_pass", "reason": reason},
        )


class APIMediaDetector(MediaDetector):
    """
    HTTP API-based detector.

    Sends a multipart/form-data POST to SYNTHETIC_MEDIA_API_URL with the
    raw content bytes.  Expects a JSON response:

        {
            "is_synthetic": bool,
            "confidence":   float,        // 0.0–1.0
            "labels":       ["GAN_face"],  // optional
            "raw":          { ... }        // optional full response
        }

    Compatible with common detector APIs (Hive Moderation, Sensity, custom).
    """

    def __init__(self, api_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._timeout = timeout

    async def detect(self, content: bytes, content_type: Optional[str] = None) -> DetectionResult:
        ct = content_type or "application/octet-stream"
        is_hf = "huggingface.co" in self._api_url or ("/" in self._api_url and not self._api_url.startswith("http"))

        # ── Preprocessing (Resize/Compress if Image is large) ────────────────
        if ct.startswith("image/") and len(content) > 512 * 1024:
            try:
                from PIL import Image
                import io
                
                old_size = len(content)
                img = Image.open(io.BytesIO(content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True) # quality 80 is better for bandwidth
                content = buf.getvalue()
                ct = "image/jpeg"
                logger.info("Preprocessed image: reduced size from %d to %d bytes", old_size, len(content))
            except Exception as e:
                logger.warning("Optional image preprocessing failed: %s", e)

        if is_hf:
            model_id = self._api_url.split("huggingface.co/models/")[-1].split("api-inference.huggingface.co/models/")[-1]
            hf_url = f"https://api-inference.huggingface.co/models/{model_id}"
            
            # Use HTTP/1.1 for Hugging Face to avoid RemoteProtocolError (HTTP/2 stream resets)
            # Disable keep-alive (max_keepalive_connections=0) to ensure fresh session on retry
            limits = httpx.Limits(max_keepalive_connections=0)
            async with httpx.AsyncClient(timeout=90.0, http1=True, http2=False, limits=limits) as client:
                headers = {
                    "Authorization":    f"Bearer {self._api_key}",
                    "Content-Type":     ct,
                    "x-wait-for-model": "true"
                }

                max_retries = 3
                resp = None
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(hf_url, content=content, headers=headers)
                        
                        if resp.status_code == 503 and attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2
                            logger.warning("HF Model loading (503), retrying in %ds... (Attempt %d/%d)", wait_time, attempt+1, max_retries)
                            await asyncio.sleep(wait_time)
                            continue
                            
                        resp.raise_for_status()
                        body = resp.json()
                        break 
                        
                    except (httpx.HTTPStatusError, httpx.RemoteProtocolError, httpx.WriteTimeout,
                            httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError,
                            httpx.ReadError, httpx.PoolTimeout) as e:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 2
                            logger.warning("HF API transient error (%s), retrying in %ds...", type(e).__name__, wait_time)
                            await asyncio.sleep(wait_time)
                        else:
                            error_body = ""
                            if resp is not None:
                                try: error_body = resp.text[:500] 
                                except: pass
                            logger.error("Hugging Face API call failed after %d retries. Last error: %s. Response: %s", 
                                         max_retries, e, error_body)
                            raise
        else:
            # Generic HTTP API support
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                headers = {"Authorization": f"Bearer {self._api_key}"}
                resp = await client.post(self._api_url, files={"file": ("content", content, ct)}, headers=headers)
                resp.raise_for_status()
                body = resp.json()

        # ── Parsing Logic ───────────────────────────────────────────────────
        if isinstance(body, list):
            prob_map = {item["label"].lower(): item["score"] for item in body if "label" in item and "score" in item}
            synth_score = sum(prob_map.get(k, 0.0) for k in ("deepfake", "synthetic", "fake", "generated"))
            if not synth_score and "realism" in prob_map:
                synth_score = 1.0 - prob_map["realism"]

            best_labels = [item["label"] for item in body if item.get("score", 0) > 0.5]
            return DetectionResult(
                detector     = f"api:{self._api_url}",
                is_synthetic = synth_score >= 0.5,
                confidence   = round(synth_score, 4),
                labels       = best_labels,
                raw_response = {"hf_client": body},
            )

        is_synthetic = bool(body.get("is_synthetic", False))
        confidence   = float(body.get("confidence", 0.0))
        labels       = list(body.get("labels", []))

        return DetectionResult(
            detector     = f"api:{self._api_url}",
            is_synthetic = is_synthetic,
            confidence   = round(confidence, 4),
            labels       = labels,
            raw_response = body,
        )

        return DetectionResult(
            detector     = f"api:{self._api_url}",
            is_synthetic = is_synthetic,
            confidence   = round(confidence, 4),
            labels       = labels,
            raw_response = body,
        )


class LocalModelMediaDetector(MediaDetector):
    """
    ONNX-based local deepfake detector — runs entirely on-device, no network.

    Loads an ONNX binary-classification model and produces a DetectionResult
    compatible with the rest of the synthetic media pipeline.

    Expected model contract
    -----------------------
    Input:  float32 tensor of shape [1, 3, H, W] (ImageNet-normalised pixels)
            Input node name configured via SYNTHETIC_MEDIA_ONNX_INPUT_NAME.
    Output: one of:
        [1, 2]  — logits [real_score, synthetic_score]  (argmax + softmax)
        [1, 1]  — sigmoid probability of synthetic       (direct confidence)

    Preprocessing
    -------------
    If the ``pillow`` (PIL) package is available, images are:
        1. Decoded from raw bytes
        2. Converted to RGB and resized to (input_size × input_size)
        3. Normalised to ImageNet mean/std ([0.485,0.456,0.406] / [0.229,0.224,0.225])
    If PIL is NOT available, a flat float32 array of the raw bytes is used
    (useful for unit testing without real images).

    Fallback
    --------
    If ``onnxruntime`` is not installed, instantiation raises a clear
    ``RuntimeError`` rather than a silent ImportError deep in the call stack.

    Labels
    ------
    The model may optionally return a ``labels`` metadata node.  If absent,
    the label is inferred from the confidence value.
    """

    # ImageNet normalisation constants
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path:  str,
        input_name:  str = "input",
        input_size:  int = 224,
    ) -> None:
        if not model_path:
            raise ValueError(
                "SYNTHETIC_MEDIA_ONNX_MODEL_PATH must be set when "
                "SYNTHETIC_MEDIA_MODE=onnx"
            )
        self._model_path = model_path
        self._input_name = input_name
        self._input_size = input_size
        self._session    = None   # lazily initialised on first call

    def _load_session(self):
        """Lazy-load the ONNX session so startup stays fast."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed.  Install it with:\n"
                "    pip install onnxruntime\n"
                "or switch SYNTHETIC_MEDIA_MODE to 'mock' or 'api'."
            ) from exc

        import os
        if not os.path.isfile(self._model_path):
            raise RuntimeError(
                f"ONNX model not found at {self._model_path!r}.  "
                "Set SYNTHETIC_MEDIA_ONNX_MODEL_PATH to a valid .onnx file."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3   # suppress verbose ort logs
        self._session = ort.InferenceSession(
            self._model_path,
            sess_options = sess_opts,
            providers    = ["CPUExecutionProvider"],
        )
        logger.info(
            "LocalModelMediaDetector: ONNX session loaded — model=%s input=%s size=%d",
            self._model_path, self._input_name, self._input_size,
        )

    def _preprocess(self, content: bytes) -> "numpy.ndarray":   # type: ignore[name-defined]
        """
        Decode and normalise image bytes to a float32 [1,3,H,W] tensor.
        Falls back to a zero-padded/truncated float32 array when PIL is absent.
        """
        import numpy as np

        size = self._input_size

        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img = img.resize((size, size))
            arr = np.array(img, dtype=np.float32) / 255.0          # [H,W,3]

            # ImageNet normalisation
            mean = np.array(self._MEAN, dtype=np.float32)
            std  = np.array(self._STD,  dtype=np.float32)
            arr  = (arr - mean) / std                               # [H,W,3]

            arr  = arr.transpose(2, 0, 1)                           # [3,H,W]
            return arr[np.newaxis, ...]                             # [1,3,H,W]

        except (ImportError, Exception):
            # PIL not installed or image unreadable — produce a deterministic
            # tensor from the content bytes (still feeds the model; only valid
            # for unit tests, not production quality detection).
            n = size * size * 3
            raw = np.frombuffer(
                (content * ((n // len(content)) + 1))[:n], dtype=np.uint8
            ).astype(np.float32) / 255.0
            return raw.reshape(1, 3, size, size)

    def _parse_output(self, output) -> tuple:
        """
        Parse ONNX output to (confidence: float, is_synthetic: bool).

        Handles two common output shapes:
          [1, 2] → logits [real_score, synthetic_score] → softmax
          [1, 1] → sigmoid probability of synthetic
        """
        import numpy as np

        arr = np.array(output).squeeze()  # collapse batch dim

        if arr.ndim == 0:
            # Scalar output — treat as synthetic probability directly
            confidence = float(arr)
        elif arr.shape == (2,):
            # [real_logit, synth_logit] — apply softmax
            exp = np.exp(arr - arr.max())
            probs = exp / exp.sum()
            confidence = float(probs[1])
        else:
            # Single value or unexpected shape — take first element
            confidence = float(arr.flat[0])

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        return round(confidence, 4), confidence >= 0.5

    async def detect(self, content: bytes, content_type: Optional[str] = None) -> DetectionResult:
        if self._session is None:
            self._load_session()

        import numpy as np

        inp   = self._preprocess(content)
        outs  = self._session.run(None, {self._input_name: inp})
        confidence, is_synthetic = self._parse_output(outs[0])

        labels = []
        if is_synthetic:
            # Derive a stable label from confidence bucket
            if confidence >= 0.9:
                labels = ["high_confidence_synthetic"]
            elif confidence >= 0.7:
                labels = ["likely_synthetic"]
            else:
                labels = ["possible_synthetic"]

        return DetectionResult(
            detector     = f"onnx:{self._model_path}",
            is_synthetic = is_synthetic,
            confidence   = confidence,
            labels       = labels,
            raw_response = {
                "engine":       "onnxruntime",
                "model_path":   self._model_path,
                "input_name":   self._input_name,
                "confidence":   confidence,
                "is_synthetic": is_synthetic,
                "labels":       labels,
            },
        )


class ConsensusDetector(MediaDetector):
    """
    Parallel consensus engine: runs the API detector chain AND the local
    heuristic simultaneously, then blends results using a determinism-first
    policy.

    Rationale:
      Probabilistic API models can be fooled by adversarial inputs and novel
      synthesis techniques.  Deterministic forensic signals (ELA, FFT, embedded
      metadata, perceptual hash cache) are mathematically derived and far harder
      to evade.  When the API says "authentic" but the math disagrees, this
      engine sides with the math — a principle adopted by industry leaders to 
      ensure high-integrity enforcement even against zero-day deepfake models.

    Blending Architecture:
      This engine implements a 'weighted consensus' model where local
      mathematical evidence acts as a 'veto' or 'lift' for the primary API result.
      If the API return value is in a low-confidence range (e.g. 0.3-0.5), 
      deterministic local signals can provide the mathematical certainty 
      necessary to trigger a BLOCK or ESCALATE action.

    Blending rule:
      If api_confidence < CONSENSUS_API_THRESHOLD
         AND deterministic_signal_sum >= DETERMINISM_FLOOR:
             # Lift the verdict: local math proves synthesis where API is uncertain.
             final = max(api_conf, local_conf * DETERMINISM_WEIGHT)
      Otherwise: API result is authoritative (API was confident).

    Compliance:
      Both results (API and Local) are preserved in raw_response for full 
      audit trail transparency. The "consensus_used" flag simplifies post-incident 
      forensics by highlighting exactly where 'math beat probability'.
    """

    _CONSENSUS_API_THRESHOLD = 0.50   # below this, API is not confident → consider local
    _DETERMINISM_FLOOR       = 0.18   # sum of deterministic signals needed to trigger override
    _DETERMINISM_WEIGHT      = 0.85   # scale factor when lifting local confidence

    # Signals derived from mathematical analysis (not probabilistic); carry override weight
    _DETERMINISTIC_KEYS: frozenset = frozenset({
        "ai_png_metadata",
        "ai_byte_signature",
        "ai_exif_software",
        "ai_tool_in_exif",
        "known_ai_hash_match",
        "ela_compression_ratio",
        "fft_grid_artifact",
        "video_frame_ai_signals",
    })

    def __init__(self, api_detector: MediaDetector, local_detector: "MockMediaDetector") -> None:
        self._api   = api_detector
        self._local = local_detector

    async def detect(self, content: bytes, content_type: Optional[str] = None) -> DetectionResult:
        # Run API and local forensics in parallel — total latency = max(api_time, local_time).
        # This parallel execution ensures that complex local heuristics like FFT/ELA 
        # do not add serial latency on top of the network call.
        api_task   = asyncio.create_task(self._api.detect(content, content_type))
        local_task = asyncio.create_task(self._local.detect(content, content_type))

        api_result:   Optional[DetectionResult] = None
        local_result: Optional[DetectionResult] = None

        try:
            api_result = await api_task
        except Exception as exc:
            logger.info("ConsensusDetector: API tier failed (%s) — using local only", exc)

        try:
            local_result = await local_task
        except Exception as exc:
            logger.warning("ConsensusDetector: Local tier failed (%s)", exc)

        # ── Degenerate cases ──────────────────────────────────────────────────
        if api_result is None and local_result is None:
            raise RuntimeError("ConsensusDetector: both API and local detectors failed")

        if api_result is None:
            raw = local_result.raw_response or {}
            raw.update({"detection_tier": "local_only", "fallback_used": True, "fallback_tier": "local_only"})
            local_result.raw_response = raw
            return local_result

        if local_result is None:
            raw = api_result.raw_response or {}
            raw["detection_tier"] = "api_only"
            api_result.raw_response = raw
            return api_result

        # ── Consensus blending ────────────────────────────────────────────────
        api_conf      = api_result.confidence
        local_conf    = local_result.confidence
        local_signals = (local_result.raw_response or {}).get("signals", {})

        # Calculate the sum of deterministic (mathematically verified) signal scores.
        # These are weights from MockMediaDetector that carry empirical mathematical proof.
        deterministic_sum = sum(
            v for k, v in local_signals.items() if k in self._DETERMINISTIC_KEYS
        )

        use_consensus = (
            api_conf < self._CONSENSUS_API_THRESHOLD and
            deterministic_sum >= self._DETERMINISM_FLOOR
        )

        if use_consensus:
            blended      = min(0.95, max(api_conf, local_conf * self._DETERMINISM_WEIGHT))
            is_synthetic = blended >= MockMediaDetector._ENFORCEMENT_THRESHOLD
            labels       = list(dict.fromkeys(api_result.labels + local_result.labels))
            raw          = {
                **(api_result.raw_response or {}),
                "detection_tier":     "consensus",
                "consensus_used":     True,
                "api_confidence":     round(api_conf, 4),
                "local_confidence":   round(local_conf, 4),
                "deterministic_sum":  round(deterministic_sum, 4),
                "blended_confidence": round(blended, 4),
                "consensus_note":     (
                    "Deterministic forensic signals exceeded API confidence — "
                    "local evidence lifted the verdict (math beats probability)."
                ),
                "local_signals":      {k: round(v, 4) for k, v in local_signals.items()},
                "signal_categories":  (local_result.raw_response or {}).get("signal_categories", {}),
            }
            logger.info(
                "ConsensusDetector: override — api=%.3f local=%.3f det_sum=%.3f → blended=%.3f",
                api_conf, local_conf, deterministic_sum, blended,
            )
            return DetectionResult(
                detector     = "consensus",
                is_synthetic = is_synthetic,
                confidence   = round(blended, 4),
                labels       = labels,
                raw_response = raw,
            )

        # API is confident or local signals are weak — API is authoritative.
        # Still enrich raw_response with local signal summary for audit trail.
        raw = api_result.raw_response or {}
        raw.update({
            "detection_tier":    "api_authoritative",
            "consensus_used":    False,
            "local_confidence":  round(local_conf, 4),
            "deterministic_sum": round(deterministic_sum, 4),
        })
        if local_signals:
            raw["local_signals"] = {k: round(v, 4) for k, v in local_signals.items()}
        api_result.raw_response = raw
        return api_result


class FallbackChainDetector(MediaDetector):
    """
    Multi-tier detector that tries a prioritised list of backends in order.

    Each tier is a (MediaDetector, str) tuple where the string is a human-readable
    label stored in raw_response["detection_tier"] for audit and compliance.

    Typical chain when SYNTHETIC_MEDIA_MODE=api with a bridge configured:
        Tier 1 → APIMediaDetector(primary)   — specialist deepfake model
        Tier 2 → APIMediaDetector(secondary) — general vision bridge
        Tier 3 → MockMediaDetector()          — local mathematical heuristics

    On complete chain exhaustion the last exception is re-raised so the
    service-level fallback in SyntheticMediaShieldService.scan() can log it.
    """

    def __init__(self, tiers: list) -> None:
        # tiers: list of (MediaDetector instance, tier_label str)
        self._tiers = tiers

    async def detect(
        self, content: bytes, content_type: Optional[str] = None
    ) -> DetectionResult:
        last_exc: Optional[Exception] = None
        for detector, tier_label in self._tiers:
            try:
                result = await detector.detect(content, content_type)
                # Tag result with which tier actually produced it
                if result.raw_response is None:
                    result.raw_response = {}
                result.raw_response["detection_tier"] = tier_label
                if tier_label != "primary_api":
                    # Let the compliance log show that a fallback was used
                    result.raw_response["fallback_used"] = True
                    result.raw_response["fallback_tier"] = tier_label
                logger.info("FallbackChain: succeeded on tier '%s'", tier_label)
                return result
            except Exception as exc:
                logger.warning(
                    "FallbackChain: tier '%s' failed (%s: %s) — trying next tier",
                    tier_label, type(exc).__name__, exc,
                )
                last_exc = exc
        raise RuntimeError(
            f"All {len(self._tiers)} detector tiers exhausted. "
            f"Last error: {last_exc}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  §3  SYNTHETIC MEDIA SHIELD SERVICE
# ══════════════════════════════════════════════════════════════════════════════

def _evidence_bundle(
    scan_id: str,
    content_hash: str,
    detection: DetectionResult,
    enforcement_action: str,
    created_at: str,
) -> dict:
    """
    Build a tamper-evident evidence bundle.

    The bundle is a canonical JSON dict whose SHA-256 hash is stored
    separately in the DB — any post-hoc modification to the stored bundle
    will be detectable by recomputing its hash.
    """
    bundle = {
        "scan_id":           scan_id,
        "content_hash":      content_hash,
        "detector":          detection.detector,
        "is_synthetic":      detection.is_synthetic,
        "confidence":        detection.confidence,
        "labels":            sorted(detection.labels),
        "enforcement_action": enforcement_action,
        "bascg_version":     "3.0",
        "created_at":        created_at,
        "nonce":             secrets.token_hex(8),  # prevents hash pre-computation
    }
    return bundle


def _hash_evidence(bundle: dict) -> str:
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Structured reason codes for URL scan INCONCLUSIVE results.
# Used by compliance officers to audit exactly why a scan could not be completed.
_URL_REASON_CODES = {
    "TIMEOUT":            "Request timed out before content could be fetched",
    "BOT_DETECTION":      "Remote server returned 403 — likely bot/scraping protection",
    "RATE_LIMITED":       "Remote server returned 429 — rate limited, retry later",
    "URL_NOT_FOUND":      "Content at this URL no longer exists (404/410)",
    "REMOTE_SERVER_ERROR":"Remote server returned a 5xx error",
    "CONNECTION_ERROR":   "Could not establish a TCP connection to the remote host",
    "PROTOCOL_ERROR":     "HTTP protocol error during content fetch (truncated response)",
    "EMPTY_CONTENT":      "URL returned an empty response body",
    "CONTENT_UNAVAILABLE":"Content is private, deleted, age-restricted, or geo-blocked",
    "EMPTY_METADATA":     "Could not extract usable thumbnail or content from this URL",
    "PARSE_ERROR":        "Response could not be parsed as a valid media file",
    "UNSUPPORTED_URL":    "URL platform is not directly supported (no direct image link)",
    "UNKNOWN_ERROR":      "An unexpected error occurred during content retrieval",
}


def _classify_url_error(exc: Exception, url: str) -> dict:
    """
    Map a URL-fetch exception to a structured INCONCLUSIVE reason code.

    Returns a dict with:
        reason_code   — machine-readable constant (e.g. "TIMEOUT")
        reason_label  — human-readable description
        error_type    — Python exception class name
        error_detail  — truncated exception message for audit log
        url           — the URL that failed
    """
    detail   = str(exc)[:300]
    exc_name = type(exc).__name__

    # Timeout variants (httpx, asyncio, yt-dlp)
    if any(t in exc_name for t in ("Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout")):
        code = "TIMEOUT"

    # HTTP status errors
    elif hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        status = exc.response.status_code
        if status == 403:
            code = "BOT_DETECTION"
        elif status == 429:
            code = "RATE_LIMITED"
        elif status in (404, 410):
            code = "URL_NOT_FOUND"
        elif status >= 500:
            code = "REMOTE_SERVER_ERROR"
        else:
            code = f"HTTP_{status}"
            if code not in _URL_REASON_CODES:
                _URL_REASON_CODES[code] = f"Remote server returned HTTP {status}"

    # Connection / network errors
    elif any(t in exc_name for t in ("ConnectError", "ConnectionError", "NetworkError", "ReadError")):
        code = "CONNECTION_ERROR"

    # Protocol / stream errors
    elif any(t in exc_name for t in ("RemoteProtocol", "ProtocolError")):
        code = "PROTOCOL_ERROR"

    # Platform-specific unavailability (yt-dlp)
    elif any(kw in detail.lower() for kw in ("unavailable", "private video", "deleted", "age-restricted", "geo")):
        code = "CONTENT_UNAVAILABLE"

    # Bot / captcha signals in response body
    elif any(kw in detail.lower() for kw in ("captcha", "forbidden", "blocked", "robot")):
        code = "BOT_DETECTION"

    # Empty / unusable content
    elif any(kw in detail.lower() for kw in ("empty", "no thumbnail", "could not fetch")):
        code = "EMPTY_CONTENT"

    # Parse / thumbnail extraction failures
    elif any(kw in detail.lower() for kw in ("thumbnail", "parse", "extract")):
        code = "EMPTY_METADATA"

    # ValueError from our own validation
    elif isinstance(exc, ValueError):
        code = "EMPTY_METADATA"

    else:
        code = "UNKNOWN_ERROR"

    return {
        "reason_code":  code,
        "reason_label": _URL_REASON_CODES.get(code, code),
        "error_type":   exc_name,
        "error_detail": detail,
        "url":          url,
    }


class SyntheticMediaShieldService:
    """
    BASCG Phase 3 — Synthetic Media Shield orchestrator.

    Responsibilities:
      1. Select detector backend from config (mock or HTTP API).
      2. Run detection on uploaded content bytes.
      3. Apply enforcement policy:
           confidence < threshold         → PASS
           confidence >= threshold        → ALERT (log)
           is_synthetic + election_context → ESCALATE (simulate ECI bus)
           explicit block list label      → BLOCK
      4. Build tamper-evident evidence bundle.
      5. Persist SyntheticMediaScanRecord to DB.
      6. Return ScanResult.
    """

    def _get_detector(self) -> MediaDetector:
        mode = getattr(settings, "SYNTHETIC_MEDIA_MODE", "mock").lower()

        if mode == "api":
            url1 = getattr(settings, "SYNTHETIC_MEDIA_API_URL", "").strip()
            key1 = getattr(settings, "SYNTHETIC_MEDIA_API_KEY", "").strip()
            if not url1:
                raise RuntimeError(
                    "SYNTHETIC_MEDIA_API_URL must be set when SYNTHETIC_MEDIA_MODE=api"
                )

            url2 = getattr(settings, "SYNTHETIC_MEDIA_API_URL_2", "").strip()
            key2 = getattr(settings, "SYNTHETIC_MEDIA_API_KEY_2", "").strip()

            # Build API-only fallback chain (bridge tier if configured)
            api_tiers = [(APIMediaDetector(api_url=url1, api_key=key1), "primary_api")]
            if url2:
                api_tiers.append((APIMediaDetector(api_url=url2, api_key=key2), "secondary_api"))
                logger.info(
                    "SyntheticMediaShield: detector → Consensus["
                    "FallbackChain(primary=%s, bridge=%s) + local_heuristic]",
                    url1, url2,
                )
            else:
                logger.info(
                    "SyntheticMediaShield: detector → Consensus[primary=%s + local_heuristic]",
                    url1,
                )

            # ConsensusDetector: API chain and local heuristic run in parallel.
            # Deterministic forensic signals can override a low-confidence API result.
            api_chain = FallbackChainDetector(api_tiers) if url2 else APIMediaDetector(api_url=url1, api_key=key1)
            return ConsensusDetector(api_detector=api_chain, local_detector=MockMediaDetector())

        if mode == "onnx":
            model_path  = getattr(settings, "SYNTHETIC_MEDIA_ONNX_MODEL_PATH", "").strip()
            input_name  = getattr(settings, "SYNTHETIC_MEDIA_ONNX_INPUT_NAME", "input")
            input_size  = int(getattr(settings, "SYNTHETIC_MEDIA_ONNX_INPUT_SIZE", 224))
            logger.info(
                "SyntheticMediaShield: detector → LocalModel/ONNX (path=%s input=%s size=%d)",
                model_path, input_name, input_size,
            )
            return LocalModelMediaDetector(
                model_path = model_path,
                input_name = input_name,
                input_size = input_size,
            )

        logger.debug("SyntheticMediaShield: detector → Mock (deterministic hash)")
        return MockMediaDetector()

    def _determine_enforcement(
        self,
        detection: DetectionResult,
        election_context: bool,
        threshold: float,
    ) -> tuple:
        """Returns (enforcement_action, policy_violations)."""
        violations = []

        # Hard-block labels — only when detection is both highly confident AND
        # carries a label that indicates identity/voice manipulation.
        # Requires confidence ≥ 0.70 to prevent speculative blocks.
        _BLOCK_LABELS = {"audio_clone", "face_swap", "lip_sync"}
        matched_block = _BLOCK_LABELS.intersection(set(detection.labels))
        if detection.is_synthetic and detection.confidence >= 0.70 and matched_block:
            violations.append({
                "policy": "synthetic-identity-block",
                "reason": f"Identity manipulation labels detected: {sorted(matched_block)}",
                "law":    "IT Act S.66E / DPDP S.4",
            })
            return "BLOCK", violations

        # Election Protection Mode — escalate
        if election_context and detection.is_synthetic and detection.confidence >= threshold:
            violations.append({
                "policy": "election-protection-mode",
                "reason": "Synthetic media detected during active election window",
                "law":    "ECI Model Code of Conduct",
            })
            return "ESCALATE", violations

        # Standard threshold alert
        if detection.confidence >= threshold:
            violations.append({
                "policy": "synthetic-media-confidence-threshold",
                "reason": f"Confidence {detection.confidence:.2%} >= threshold {threshold:.2%}",
                "law":    "BASCG Phase 3 P3 policy",
            })
            return "ALERT", violations

        return "PASS", violations

    async def scan(
        self,
        content: bytes,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
        submitted_by: Optional[str] = None,
        source_ip: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> ScanResult:
        """
        Run a full synthetic media scan.

        Args:
            content:      Raw bytes of the file to scan.
            content_type: MIME type hint (e.g. "image/jpeg").
            filename:     Original filename (stored sanitised, not used for detection).
            submitted_by: Username or API key of caller.
            source_ip:    Client IP for audit.
            db:           Optional AsyncSession — if provided, persists result to DB.

        Returns:
            ScanResult with enforcement decision and evidence bundle.
        """
        from app.models.orm_models import SyntheticMediaScanRecord

        scan_id      = secrets.token_hex(16)
        content_hash = hashlib.sha256(content).hexdigest()
        now_iso      = datetime.now(timezone.utc).isoformat()

        # Run detection
        detector  = self._get_detector()
        try:
            detection = await detector.detect(content, content_type)
        except Exception as e:
            # Persistent failure — Fallback to Mock Detector to ensure system availability
            logger.error("Persistent detector failure (%s): %s. Falling back to Mock Detector.", 
                         type(detector).__name__, e)
            fallback_detector = MockMediaDetector() # We are in the same module
            detection = await fallback_detector.detect(content, content_type)
            detection.verdict = f"FALLBACK: {str(e)[:100]}"

        # Config — Default threshold to 0.45 (Aggressive Governance)
        # This ensures consistency: if probability > 45%, it's marked as Synthetic (BLOCK/ALERT).
        threshold        = float(getattr(settings, "SYNTHETIC_MEDIA_CONFIDENCE_THRESHOLD", 0.45))
        epm_enabled      = bool(getattr(settings, "ELECTION_PROTECTION_ENABLED", False))
        election_state   = getattr(settings, "ELECTION_PROTECTION_STATE", None)
        election_context = epm_enabled and bool(election_state)
        
        # Ensure deepfake flag is strictly consistent with the active threshold (0.45)
        # This prevents "Authentic" labels when probability is above the limit.
        if detection.confidence >= threshold:
            detection.is_synthetic = True
        else:
            detection.is_synthetic = False

        # Enforcement
        enforcement_action, policy_violations = self._determine_enforcement(
            detection, election_context, threshold
        )

        # Escalation (Election Protection Mode)
        escalated = False
        if enforcement_action == "ESCALATE":
            from app.services.eci_webhook_service import eci_webhook_service
            eci_result = await eci_webhook_service.escalate(
                scan_id       = scan_id,
                state         = election_state or "",
                confidence    = detection.confidence,
                labels        = detection.labels,
                evidence_hash = content_hash,   # SHA-256 of raw content bytes
                detector      = detection.detector,
            )
            escalated = eci_result.sent or eci_result.stub

        # Evidence bundle
        bundle      = _evidence_bundle(scan_id, content_hash, detection, enforcement_action, now_iso)
        ev_hash     = _hash_evidence(bundle)

        result = ScanResult(
            scan_id            = scan_id,
            content_hash       = content_hash,
            content_type       = content_type,
            file_size_bytes    = len(content),
            filename           = filename,
            detection          = detection,
            enforcement_action = enforcement_action,
            policy_violations  = policy_violations,
            election_context   = election_context,
            election_state     = election_state if election_context else None,
            escalated_to_eci   = escalated,
            evidence_hash      = ev_hash,
            evidence_bundle    = bundle,
            submitted_by       = submitted_by,
            source_ip          = source_ip,
            created_at         = now_iso,
        )

        # Persist
        if db is not None:
            record = SyntheticMediaScanRecord(
                id                 = scan_id,
                content_hash       = content_hash,
                content_type       = content_type,
                file_size_bytes    = len(content),
                filename           = filename,
                detector           = detection.detector,
                is_synthetic       = detection.is_synthetic,
                confidence         = detection.confidence,
                detection_labels   = detection.labels,
                raw_response       = detection.raw_response,
                enforcement_action = enforcement_action,
                policy_violations  = policy_violations,
                election_context   = election_context,
                election_state     = election_state if election_context else None,
                escalated_to_eci   = escalated,
                evidence_hash      = ev_hash,
                evidence_bundle    = bundle,
                submitted_by       = submitted_by,
                source_ip          = source_ip,
            )
            db.add(record)
            await db.commit()
            logger.info(
                "SyntheticMediaShield: scan_id=%s action=%s synthetic=%s conf=%.2f",
                scan_id, enforcement_action, detection.is_synthetic, detection.confidence,
            )

        return result

    async def scan_url(
        self,
        url: str,
        submitted_by: Optional[str] = None,
        source_ip: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> ScanResult:
        """
        Fetch content from a URL (webpage, YouTube, Instagram, etc.) and scan it.
        For video platforms, we extract a representative thumbnail/frame for image-based
        deepfake analysis — the SigLIP-2 model only accepts images.
        """
        import re as _re

        lower_url = url.lower()
        filename: str = url.split("/")[-1].split("?")[0] or "url_scan"

        ctx_content: bytes = b""
        ctx_type: str = "image/jpeg"

        try:
            # ── YouTube: Direct thumbnail fetch (no yt-dlp needed) ────────────
            yt_match = _re.search(r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})', url)
            if yt_match:
                video_id = yt_match.group(1)
                logger.info("YouTube video detected (ID=%s). Fetching thumbnail directly.", video_id)

                # Try HD thumbnail first, fall back to standard
                thumb_urls = [
                    f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                    f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
                    f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                ]

                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    for thumb_url in thumb_urls:
                        try:
                            resp = await client.get(thumb_url)
                            # YouTube returns a tiny grey placeholder for missing thumbs
                            if resp.status_code == 200 and len(resp.content) > 2000:
                                ctx_content = resp.content
                                ctx_type = "image/jpeg"
                                filename = f"YT:{video_id}"
                                logger.info("Got YouTube thumbnail (%d bytes) from %s", len(ctx_content), thumb_url)
                                break
                        except Exception:
                            continue

                if not ctx_content:
                    raise ValueError(f"Could not fetch thumbnail for YouTube video {video_id}")

            # ── Other video platforms (Instagram, TikTok, etc.): yt-dlp ───────
            elif any(k in lower_url for k in ["instagram.com", "tiktok.com", "facebook.com", "reels"]):
                logger.info("Social media URL detected, using yt-dlp: %s", url)
                ydl_opts = {
                    'format': 'best[height<=720]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'skip_download': True,
                    'nocheckcertificate': True,
                }

                def fetch_info(v_url):
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        return ydl.extract_info(v_url, download=False)

                info = await asyncio.to_thread(fetch_info, url)
                thumbnail_url = info.get('thumbnail')
                if not thumbnail_url:
                    raise ValueError("No thumbnail found in yt-dlp metadata")

                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(thumbnail_url)
                    resp.raise_for_status()
                    ctx_content = resp.content
                    ctx_type = resp.headers.get("Content-Type", "image/jpeg")
                    filename = f"Social:{url.split('/')[2][:20]}"

            # ── Standard URL: direct fetch (images, etc.) ─────────────────────
            else:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    ctx_content = resp.content
                    ctx_type = resp.headers.get("Content-Type", "application/octet-stream")

            if not ctx_content:
                raise ValueError("Fetched content is empty")

            return await self.scan(
                content=ctx_content,
                content_type=ctx_type,
                filename=f"URL: {filename[:30]}",
                submitted_by=submitted_by,
                source_ip=source_ip,
                db=db
            )

        except Exception as e:
            reason  = _classify_url_error(e, url)
            code    = reason["reason_code"]
            label   = reason["reason_label"]
            logger.warning(
                "URL content fetch failed for %s — reason_code=%s (%s: %s)",
                url, code, reason["error_type"], reason["error_detail"][:120],
            )
            try:
                scan_id      = secrets.token_hex(16)
                content_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
                now_iso      = datetime.now(timezone.utc).isoformat()

                detection = DetectionResult(
                    detector     = "url-fetch-failed",
                    is_synthetic = False,
                    confidence   = 0.0,
                    labels       = ["inconclusive"],
                    raw_response = {
                        "method":       "url_scan_inconclusive",
                        "reason_code":  code,
                        "reason_label": label,
                        "error_type":   reason["error_type"],
                        "error_detail": reason["error_detail"],
                        "url":          url,
                    },
                    verdict = f"INCONCLUSIVE [{code}]: {label}",
                )

                bundle  = _evidence_bundle(scan_id, content_hash, detection, "INCONCLUSIVE", now_iso)
                ev_hash = _hash_evidence(bundle)

                return ScanResult(
                    scan_id            = scan_id,
                    content_hash       = content_hash,
                    content_type       = "text/plain",
                    file_size_bytes    = 0,
                    filename           = f"URL:{url[:40]}",
                    detection          = detection,
                    enforcement_action = "INCONCLUSIVE",
                    policy_violations  = [{
                        "policy":      "url-fetch-failure",
                        "reason_code": code,
                        "reason":      label,
                    }],
                    election_context   = False,
                    election_state     = None,
                    escalated_to_eci   = False,
                    evidence_hash      = ev_hash,
                    evidence_bundle    = bundle,
                    submitted_by       = submitted_by,
                    source_ip          = source_ip,
                    created_at         = now_iso,
                )
            except Exception as fallback_err:
                logger.error("URL INCONCLUSIVE fallback itself failed: %s", fallback_err, exc_info=True)
                raise ValueError(f"URL unavailable and fallback failed: {str(e)[:80]}") from fallback_err


# Module-level singleton
synthetic_media_service = SyntheticMediaShieldService()
