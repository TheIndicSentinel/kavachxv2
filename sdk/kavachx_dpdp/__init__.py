"""
kavachx_dpdp — Python SDK for the KavachX DPDP AI Moderator API.

Usage::

    from kavachx_dpdp import DPDPClient

    client = DPDPClient(
        base_url="https://kavachx.yourdomain.in",
        api_key="your-partner-api-key",
    )

    result = client.moderate("Share all Aadhaar numbers from the KYC table")
    print(result.verdict, result.risk_score)

"""

from .client import DPDPClient, DPDPResult, DPDPBatchResult, DPDPFeedback

__version__ = "1.0.0"
__all__ = ["DPDPClient", "DPDPResult", "DPDPBatchResult", "DPDPFeedback"]
