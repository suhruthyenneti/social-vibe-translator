"""
Social Vibe Translator - FastAPI + MCP server entrypoint.

This server exposes a FastAPI HTTP endpoint `/rewrite_vibes` and an MCP tool
named `rewrite_vibes` for Model Context Protocol clients. It uses OpenAI for
generation when available and falls back to local heuristics otherwise.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tone_analyzer import analyze_tone
from vibe_generator import generate_vibes
from platform_advisor import get_platform_tips
from utils import load_environment
from judge_rerank import rank_rewrites
from rag.store import seed_guidelines, upsert_user_example
from moderation import mask_pii, moderate_text

# Try to import MCP SDK (model context protocol). Keep optional for HTTP-only usage.
try:
    from mcp.server.fastapi import FastAPIHandler
    from mcp.types import (Tool, ToolInputSchema, ToolParameter,
                           CallToolRequest)
    MCP_AVAILABLE = True
except Exception:  # pragma: no cover
    MCP_AVAILABLE = False


# ----------------------------- Pydantic Models ----------------------------- #


class VibeItem(BaseModel):
    """A single vibe rewrite output item."""

    vibe: str
    rewritten_text: str
    explanation: str
    use_cases: List[str]


class ToneResult(BaseModel):
    """Tone analysis result for the input message."""

    overall_tone: str
    rationale: str


class RewriteVibesRequest(BaseModel):
    """Input schema for the rewrite tool/API."""

    message: str = Field(..., description="Original message to rewrite.")
    platform: Optional[str] = Field(None, description="Optional platform context (e.g., WhatsApp, LinkedIn, Email)")


class RewriteVibesResponse(BaseModel):
    """Response schema including tone, five vibes, and platform tips."""

    original_message: str
    tone_analysis: ToneResult
    vibes: List[VibeItem]
    platform_tips: Dict[str, str]


class FeedbackRequest(BaseModel):
    """User feedback to store an accepted rewrite as a style example for RAG."""

    user_id: str
    message: str
    accepted_text: str
    platform: Optional[str] = None
    target_tone: str = Field(..., description="Tone of the accepted rewrite")


class RewriteTopRequest(BaseModel):
    """Input for top-3 rewrite candidates for a chosen tone."""

    message: str
    platform: Optional[str] = None
    target_tone: str = Field(..., description="Desired tone for the rewrite, e.g., Professional/Friendly/etc.")
    num_candidates: int = Field(3, ge=1, le=10, description="How many top candidates to return (default 3)")


class RankedCandidate(BaseModel):
    """A ranked rewrite with score."""

    vibe: str
    rewritten_text: str
    explanation: str
    use_cases: List[str]
    score: float


class RewriteTopResponse(BaseModel):
    """Response for top-N ranked rewrites for a single target tone."""

    original_message: str
    target_tone: str
    platform_tips: Dict[str, str]
    top_rewrites: List[RankedCandidate]


# ------------------------------- FastAPI App ------------------------------- #


load_environment()
app = FastAPI(title="Social Vibe Translator", version="0.1.0")


@app.post("/rewrite_vibes", response_model=RewriteVibesResponse)
async def rewrite_vibes_api(payload: RewriteVibesRequest) -> RewriteVibesResponse:
    """HTTP endpoint to analyze and rewrite a message into five vibes."""

    # Light moderation and PII masking
    _ = await moderate_text(payload.message)
    clean_message = mask_pii(payload.message)

    tone = await analyze_tone(clean_message)
    vibes = await generate_vibes(clean_message, platform=payload.platform)
    tips = get_platform_tips(payload.platform)

    return RewriteVibesResponse(
        original_message=payload.message,
        tone_analysis=ToneResult(**tone),
        vibes=[VibeItem(**v) for v in vibes],
        platform_tips=tips,
    )


@app.post("/rewrite_top", response_model=RewriteTopResponse)
async def rewrite_top_api(payload: RewriteTopRequest) -> RewriteTopResponse:
    """Return top-N ranked rewrites for a chosen target tone.

    We reuse the 5 vibes generator, then filter/select candidates whose vibe
    matches the `target_tone` or is closest by name. If not found, we use
    all candidates and let the ranker decide.
    """

    _ = await moderate_text(payload.message)
    clean_message = mask_pii(payload.message)
    all_candidates = await generate_vibes(clean_message, platform=payload.platform)

    # Use all candidates and let the ranker choose best matches to target_tone
    candidates = all_candidates

    scored = await rank_rewrites(
        candidates=candidates,
        message=clean_message,
        target_tone=payload.target_tone,
        platform=payload.platform,
    )
    # Sort by score desc and take top N
    topn = sorted(scored, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: payload.num_candidates]
    tips = get_platform_tips(payload.platform)

    return RewriteTopResponse(
        original_message=clean_message,
        target_tone=payload.target_tone,
        platform_tips=tips,
        top_rewrites=[RankedCandidate(**c) for c in topn],
    )


# --------------------------- MCP Tool Registration ------------------------- #


if MCP_AVAILABLE:
    handler = FastAPIHandler(app)

    # Define the tool schema for MCP discovery
    rewrite_tool = Tool(
        name="rewrite_vibes",
        description=(
            "Analyze tone and rewrite a message in five vibes: Professional, Friendly, "
            "Persuasive, Concise, Empathetic. Also returns platform tips."
        ),
        inputSchema=ToolInputSchema(
            type="object",
            properties={
                "message": {"type": "string"},
                "platform": {"type": ["string", "null"], "nullable": True},
            },
            required=["message"],
        ),
    )

    @handler.tool(rewrite_tool)
    async def rewrite_vibes_tool(request: CallToolRequest) -> str:
        """MCP tool implementation that mirrors the HTTP endpoint behavior.

        Returns a JSON string compatible with MCP tool responses.
        """

        params = request.arguments or {}
        message = str(params.get("message", ""))
        platform = params.get("platform")
        if not message:
            return json.dumps({"error": "message is required"})

        _ = await moderate_text(message)
        clean_message = mask_pii(message)

        tone = await analyze_tone(clean_message)
        vibes = await generate_vibes(clean_message, platform=platform)
        tips = get_platform_tips(platform)

        response = RewriteVibesResponse(
            original_message=clean_message,
            tone_analysis=ToneResult(**tone),
            vibes=[VibeItem(**v) for v in vibes],
            platform_tips=tips,
        )
        return response.json()

    # Additional MCP tool for top-ranked rewrites
    rewrite_top_tool = Tool(
        name="rewrite_top",
        description="Return top N ranked rewrites for a chosen target tone.",
        inputSchema=ToolInputSchema(
            type="object",
            properties={
                "message": {"type": "string"},
                "platform": {"type": ["string", "null"], "nullable": True},
                "target_tone": {"type": "string"},
                "num_candidates": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            required=["message", "target_tone"],
        ),
    )

    @handler.tool(rewrite_top_tool)
    async def rewrite_top_tool_impl(request: CallToolRequest) -> str:
        params = request.arguments or {}
        message = str(params.get("message", ""))
        platform = params.get("platform")
        target_tone = str(params.get("target_tone", ""))
        num_candidates = int(params.get("num_candidates", 3))
        if not message or not target_tone:
            return json.dumps({"error": "message and target_tone are required"})

        _ = await moderate_text(message)
        clean_message = mask_pii(message)
        all_candidates = await generate_vibes(clean_message, platform=platform)
        # Use all candidates and let the ranker choose best matches to target_tone
        candidates = all_candidates

        scored = await rank_rewrites(
            candidates=candidates,
            message=clean_message,
            target_tone=target_tone,
            platform=platform,
        )
        topn = sorted(scored, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: num_candidates]
        tips = get_platform_tips(platform)

        payload = RewriteTopResponse(
            original_message=clean_message,
            target_tone=target_tone,
            platform_tips=tips,
            top_rewrites=[RankedCandidate(**c) for c in topn],
        )
        return payload.json()

    # Expose handler.router as the MCP path if needed by hosting platform
    # Users can run stdio server via: `python -m <module> --stdio`

    # Optional: stdio entry if invoked directly
    async def _maybe_run_stdio():  # pragma: no cover
        import argparse
        parser = argparse.ArgumentParser(description="Social Vibe Translator MCP Server")
        parser.add_argument("--stdio", action="store_true", help="Run MCP stdio server")
        args, _ = parser.parse_known_args()
        if args.stdio:
            await handler.run_stdio()

    # Schedule stdio runner if requested via env flag
    if os.getenv("MCP_STDIO", "0") == "1":  # pragma: no cover
        asyncio.get_event_loop().create_task(_maybe_run_stdio())


@app.post("/seed_guidelines")
async def seed_guidelines_endpoint() -> Dict[str, int]:
    """Seed the RAG store with default platform/tone guidelines."""
    count = seed_guidelines()
    return {"inserted": count}


@app.post("/feedback_accept")
async def feedback_accept(payload: FeedbackRequest) -> Dict[str, str]:
    """Store an accepted rewrite as a user example for future personalization."""

    doc_id = upsert_user_example(
        user_id=payload.user_id,
        message=payload.message,
        platform=(payload.platform or "generic").lower(),
        target_tone=payload.target_tone,
        accepted_text=payload.accepted_text,
    )
    return {"stored": doc_id}


@app.get("/test_gemini")
async def test_gemini() -> Dict[str, str]:
    """Test if Gemini generation is working."""
    from utils import gemini_json_completion
    try:
        result = await gemini_json_completion(
            system_prompt="Return JSON with field 'test' containing a greeting",
            user_prompt="Say hello"
        )
        return {"status": "success", "result": str(result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/test_vibes")
async def test_vibes() -> Dict[str, Any]:
    """Test vibe generation directly with debug info."""
    from utils import get_openai_client, openai_json_completion
    from config.vibe_templates import VIBE_TEMPLATES
    from rag.store import retrieve_docs
    
    try:
        # Direct test of the AI generation step
        text = "Hello world"
        client = get_openai_client()
        retrieved = retrieve_docs(query=f"linkedin guidance for: {text}", platform="linkedin", user_id=None, top_k=5)
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
        
        ai_result = await openai_json_completion(client=client, system_prompt=system, user_prompt=user)
        
        return {
            "status": "debug",
            "client_available": client is not None,
            "ai_result_type": str(type(ai_result)),
            "ai_result_len": len(ai_result) if isinstance(ai_result, list) else "N/A",
            "ai_result": ai_result,
            "retrieved_count": len(retrieved),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

__all__ = [
    "app",
    "RewriteVibesRequest",
    "RewriteVibesResponse",
]


