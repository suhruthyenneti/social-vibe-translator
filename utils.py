"""
Utility helpers for the Social Vibe Translator project.

This module centralizes environment loading, OpenAI client setup,
and common JSON-safe OpenAI generation helpers.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    # Prefer the async client for non-blocking FastAPI endpoints
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - keep import safe for environments without openai
    AsyncOpenAI = None  # type: ignore

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore


def load_environment() -> None:
    """Load environment variables from a .env file if present.

    This allows the developer to place `OPENAI_API_KEY` (and other keys)
    in a `.env` file for local development without exporting them in shell profile.
    """

    load_dotenv()


def get_openai_client() -> Optional["AsyncOpenAI"]:
    """Return an initialized OpenAI client if API key is available.

    Returns None if the OpenAI SDK is not installed or the API key is missing.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if AsyncOpenAI is None or not api_key:
        return None
    return AsyncOpenAI(api_key=api_key)


def safe_json_parse(text: str) -> Any:
    """Safely parse JSON text, trimming code fences if present.

    Many LLM responses may wrap JSON in triple backticks; this helper removes
    those fences and attempts to parse the inner JSON string.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove the first fence line
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        # Remove a trailing closing fence if present
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return text


async def gemini_json_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.7,
) -> Any:
    """Call Gemini API and try to parse the response as JSON."""
    
    if genai is None:
        raise ValueError("google-generativeai is not available. Install google-generativeai package.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    
    genai.configure(api_key=api_key)
    
    # Combine system and user prompts for Gemini
    full_prompt = f"{system_prompt}\n\nUser request: {user_prompt}\n\nRespond with valid JSON only:"
    
    model_obj = genai.GenerativeModel(model)
    response = await asyncio.to_thread(
        model_obj.generate_content,
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            candidate_count=1,
        )
    )
    
    content = response.text if response.text else "{}"
    return safe_json_parse(content)


async def openai_json_completion(
    *,
    client: Optional["AsyncOpenAI"],
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Any:
    """Call OpenAI Chat Completions API and try to parse the response as JSON.

    If the client is None (no key or SDK), tries Gemini fallback first.
    """

    # Try Gemini first if OpenAI is not available
    if client is None:
        try:
            return await gemini_json_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception:
            raise ValueError("Neither OpenAI nor Gemini clients are available.")

    completion = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )  # type: ignore[call-arg]

    content = completion.choices[0].message.content if completion.choices else "{}"
    if content is None:
        content = "{}"
    return safe_json_parse(content)


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate a string to a maximum number of characters.

    This keeps prompts small and avoids excessive tokens.
    """

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning a default on failure."""

    try:
        return float(value)
    except Exception:
        return default


