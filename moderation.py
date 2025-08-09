"""
Basic moderation and PII masking utilities.

For MVP: use OpenAI moderation if available; otherwise, pass-through with
lightweight masking of emails and phone numbers.
"""

from __future__ import annotations

import re
from typing import Dict

from utils import get_openai_client


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")


def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    return text


async def moderate_text(text: str) -> Dict[str, str]:
    client = get_openai_client()
    if client is None:
        return {"flagged": "no", "reason": "no_moderation_client"}

    try:
        # Using text moderation endpoint via chat is more complex; for MVP we skip
        # and assume pass unless explicit policy is integrated.
        # Placeholder so callers can branch on it.
        return {"flagged": "no", "reason": "not_implemented"}
    except Exception:
        return {"flagged": "no", "reason": "moderation_error"}


