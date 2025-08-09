"""
Tone analysis for the Social Vibe Translator.

For MVP simplicity, we provide a lightweight heuristic tone analysis and
optionally enhance it with OpenAI if an API key is available.
"""

from __future__ import annotations

from typing import Dict

from utils import get_openai_client, openai_json_completion, truncate_text


async def analyze_tone(message: str) -> Dict[str, str]:
    """Analyze tone of the given message.

    Returns a dictionary with fields like `overall_tone` and `rationale`.
    Uses OpenAI if available; otherwise falls back to a heuristic.
    """

    text = truncate_text(message, 2000)
    client = get_openai_client()

    system = (
        "You analyze the tone of short user messages."
        " Return strict JSON with keys: overall_tone (string), rationale (string)."
    )
    user = (
        "Analyze the tone of this message and return JSON only.\n\n"
        f"Message: {text}\n"
    )

    try:
        result = await openai_json_completion(client=client, system_prompt=system, user_prompt=user)
        if isinstance(result, dict) and "overall_tone" in result:
            return {"overall_tone": str(result.get("overall_tone", "Unknown")), "rationale": str(result.get("rationale", ""))}
    except Exception:
        # If OpenAI is unavailable, fall back to heuristic
        pass

    # Simple heuristic fallback
    lowered = message.lower()
    if any(w in lowered for w in ["please", "would you", "kindly", "appreciate"]):
        tone = "Polite"
    elif any(w in lowered for w in ["urgent", "asap", "now", "immediately"]):
        tone = "Urgent"
    elif any(w in lowered for w in ["sorry", "apologize", "regret"]):
        tone = "Apologetic"
    elif any(w in lowered for w in ["great", "awesome", "thanks", "thank you"]):
        tone = "Positive"
    else:
        tone = "Neutral"

    return {
        "overall_tone": tone,
        "rationale": "Heuristic analysis based on presence of polite, urgent, apologetic, or positive keywords.",
    }


