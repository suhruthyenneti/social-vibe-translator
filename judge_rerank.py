"""
Judge & rerank utilities for selecting the best rewrites.

This module optionally uses OpenAI to score candidates by a rubric and
falls back to a simple heuristic if the model isn't available.
"""

from __future__ import annotations

from typing import Any, Dict, List

from utils import get_openai_client, openai_json_completion, gemini_json_completion, to_float


async def rank_rewrites(
    *,
    candidates: List[Dict[str, Any]],
    message: str,
    target_tone: str,
    platform: str | None,
) -> List[Dict[str, Any]]:
    """Return the same candidates with an added float `score` field.

    If OpenAI is available, ask for a 0-10 score for each candidate based on:
    - Tone alignment with `target_tone`
    - Clarity and readability
    - Platform fit (format, length, conventions)

    Otherwise, use a simple heuristic that favors concise text and basic tone keywords.
    """

    client = get_openai_client()
    if client is not None and candidates:
        # Ask the model to score each candidate in order
        system = (
            "You are a precise evaluator. Score each candidate (0-10) based on:"
            " 1) Tone alignment to the requested tone,"
            " 2) Clarity and readability,"
            " 3) Fit for the specified platform."
            " Return STRICT JSON array of numbers, one per candidate, same order."
        )
        # Keep just the text to reduce token usage
        texts = [c.get("rewritten_text", "") for c in candidates]
        platform_str = platform or "generic"
        user = (
            f"Target tone: {target_tone}\nPlatform: {platform_str}\n\n"
            f"Original message: {message}\n\n"
            "Candidates (score each in order):\n" + "\n".join([f"- {t}" for t in texts])
        )
        try:
            scores = await openai_json_completion(client=client, system_prompt=system, user_prompt=user)
            if isinstance(scores, list) and len(scores) == len(candidates):
                for i, s in enumerate(scores):
                    candidates[i]["score"] = to_float(s, 0.0)
                return candidates
        except Exception:
            pass
    
    # Try Gemini if OpenAI failed or unavailable
    try:
        scores = await gemini_json_completion(system_prompt=system, user_prompt=user)
        if isinstance(scores, list) and len(scores) == len(candidates):
            for i, s in enumerate(scores):
                candidates[i]["score"] = to_float(s, 0.0)
            return candidates
    except Exception:
        pass

    # Heuristic fallback
    def heuristic_score(text: str) -> float:
        length = len(text)
        # Prefer 100-350 chars for most platforms
        if length <= 40:
            base = 4.0
        elif length <= 100:
            base = 7.0
        elif length <= 350:
            base = 8.5
        elif length <= 700:
            base = 7.2
        else:
            base = 5.5

        tone = target_tone.lower()
        txt = text.lower()
        bonus = 0.0
        if tone in ("professional", "formal") and any(k in txt for k in ["regards", "sincerely", "appreciate"]):
            bonus += 0.5
        if tone in ("friendly",) and any(k in txt for k in ["thanks", "excited", "glad", "hey"]):
            bonus += 0.5
        if tone in ("concise",) and length < 200:
            bonus += 0.5
        if tone in ("persuasive",) and any(k in txt for k in ["benefit", "impact", "value", "recommend"]):
            bonus += 0.5
        if tone in ("empathetic",) and any(k in txt for k in ["understand", "appreciate", "support", "sorry"]):
            bonus += 0.5

        return base + bonus

    for c in candidates:
        c["score"] = heuristic_score(str(c.get("rewritten_text", "")))
    return candidates


