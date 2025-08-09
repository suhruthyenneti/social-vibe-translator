"""
Vibe generator for rewriting messages into five predefined styles.

This module calls OpenAI when available to produce rewrites; otherwise,
it falls back to deterministic template-based rewrites for hackathon-friendly
local testing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config.vibe_templates import VIBE_TEMPLATES
from utils import get_openai_client, openai_json_completion, truncate_text
from rag.store import retrieve_docs
from validators import validate_platform


async def generate_vibes(message: str, *, platform: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, object]]:
    """Generate five rewrite variations for the message.

    Each item includes: vibe, rewritten_text, explanation, use_cases (list of strings).
    """

    text = truncate_text(message, 2000)
    client = get_openai_client()

    # We'll ask the model to return JSON for all five vibes in one shot
    retrieved = retrieve_docs(query=f"{platform or 'generic'} guidance for: {text}", platform=platform, user_id=user_id, top_k=5)
    grounding = "\n\nRetrieved guidance:\n" + "\n".join([f"- {r['title']}: {r['text'][:240]}" for r in retrieved]) if retrieved else ""

    system = (
        "You rewrite short messages in multiple specific tones."
        " Return strict JSON array with exactly 5 objects, each having keys:"
        " vibe, rewritten_text, explanation, use_cases (array of short strings)."
        " The five vibes must be: Professional, Friendly, Persuasive, Concise, Empathetic."
    )
    vibe_instructions = "\n".join([f"- {name}: {prompt[:180]}..." for name, prompt in VIBE_TEMPLATES.items()])
    user = (
        "Rewrite the message into five vibes using the guidance below, respond with JSON only.\n\n"
        f"Message: {text}\n\nVibe guidance:\n{vibe_instructions}\n"
        f"{grounding}"
    )

    try:
        result = await openai_json_completion(client=client, system_prompt=system, user_prompt=user)
        print(f"DEBUG: Generated result type={type(result)}, len={len(result) if isinstance(result, list) else 'N/A'}, content={result}")
        if isinstance(result, list) and len(result) == 5:
            # Basic normalization
            normalized = []
            for item in result:
                text = str(item.get("rewritten_text", ""))
                if platform:
                    vp = validate_platform(text, platform)
                    text = vp["text"]
                normalized.append({
                    "vibe": str(item.get("vibe", "")),
                    "rewritten_text": text,
                    "explanation": str(item.get("explanation", "")),
                    "use_cases": [str(u) for u in item.get("use_cases", [])][:4],
                })
            return normalized
        else:
            print(f"DEBUG: Result format mismatch - expected list of 5, got {type(result)} with len {len(result) if isinstance(result, list) else 'N/A'}")
    except Exception as e:
        # Log the error for debugging
        print(f"Vibe generation error: {e}")
        pass

    # Local deterministic fallback for offline demos
    rewrites = []
    for vibe, template in VIBE_TEMPLATES.items():
        rewritten = f"[{vibe}] {text}"
        explanation = f"Uses {vibe.lower()} tone cues based on simple template guidance."
        use_cases = [
            f"Use when you need a {vibe.lower()} tone.",
            "Useful for quick edits when time is limited.",
        ]
        rewrites.append(
            {
                "vibe": vibe,
                "rewritten_text": rewritten,
                "explanation": explanation,
                "use_cases": use_cases,
            }
        )

    # Ensure we always return exactly in the standard order
    order = ["Professional", "Friendly", "Persuasive", "Concise", "Empathetic"]
    by_name = {r["vibe"]: r for r in rewrites}
    return [by_name[name] for name in order]


