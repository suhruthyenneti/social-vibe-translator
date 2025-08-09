"""
Platform advisor provides platform-specific messaging tips.

The logic is intentionally simple and static for MVP reliability.
"""

from __future__ import annotations

from typing import Dict


PLATFORM_TIPS: Dict[str, str] = {
    "whatsapp": "Keep it short, use line breaks for readability. Emojis help convey tone, but don't overuse.",
    "linkedin": "Stay professional, avoid slang, include a clear ask, and keep paragraphs short.",
    "email": "Use a clear subject, polite greeting, one key ask, and a short signature block.",
    "twitter": "Be concise and action-oriented; consider a thread for longer thoughts.",
    "sms": "Very concise, one clear ask, avoid links unless necessary.",
}


def get_platform_tips(platform: str | None) -> Dict[str, str]:
    """Return platform-specific tips and a normalized platform name.

    If the platform is None or unknown, provide a generic tip.
    """

    if not platform:
        return {
            "platform": "generic",
            "tips": "Adapt tone to the audience; keep it clear, short, and respectful.",
        }

    key = platform.strip().lower()
    tip = PLATFORM_TIPS.get(key)
    if tip:
        return {"platform": key, "tips": tip}
    return {
        "platform": key,
        "tips": "No specific guidance found; keep it concise and audience-appropriate.",
    }


