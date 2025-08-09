## Social Vibe Translator

Beginner-friendly FastAPI + MCP server that analyzes tone and rewrites a message into five "vibes": Professional, Friendly, Persuasive, Concise, Empathetic. It also provides platform-specific tips.

### Features
- MCP tool `rewrite_vibes` with JSON I/O
- FastAPI endpoint `/rewrite_vibes`
- OpenAI-backed generation with offline fallbacks
- Retrieval-Augmented Generation (RAG) with Chroma for platform/user grounding
- Structured JSON outputs (OpenAI JSON mode)
- Ranking with LLM-as-judge fallback to heuristic
- Pydantic models for request/response
- Clear prompts and simple templates

### Requirements
- Python 3.10+
- An OpenAI API key (optional; falls back to deterministic output if missing)

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
# Set to 1 to auto-start MCP stdio mode when running server module directly
MCP_STDIO=0
```

### Run the FastAPI server
```bash
uvicorn server:app --reload --app-dir .
```
Server runs at `http://127.0.0.1:8000`.

### Seed RAG guidelines (once)
```bash
curl -X POST http://127.0.0.1:8000/seed_guidelines
```

### HTTP usage example
POST `http://127.0.0.1:8000/rewrite_vibes`
```bash
curl -s -X POST http://127.0.0.1:8000/rewrite_vibes \
  -H 'Content-Type: application/json' \
  -d '{"message": "Can we move the meeting to tomorrow?", "platform": "Email"}' | jq .
```

### MCP server usage
This project exposes an MCP tool named `rewrite_vibes` via stdio. You can run stdio mode by:
```bash
python -m server --stdio
```
If `MCP_STDIO=1` is set in `.env`, the stdio server will auto-start when the module is imported.

Register this MCP server in Puch AI:
- Tool name: `rewrite_vibes`
- Command: `python -m server --stdio`

### Request/Response schema
- Input JSON: `{ "message": str, "platform": Optional[str] }`
- Output JSON fields:
  - `original_message`: str
  - `tone_analysis`: `{ overall_tone: str, rationale: str }`
  - `vibes`: Array of 5 items, each `{ vibe, rewritten_text, explanation, use_cases }`
  - `platform_tips`: `{ platform: str, tips: str }`

Additional endpoints:
- `POST /rewrite_top` → top N ranked rewrites for `target_tone`
- `POST /seed_guidelines` → seed platform/tone guidelines into vector store
- `POST /feedback_accept` → store accepted rewrites for personalization RAG

### Example request/response
Request:
```json
{
  "message": "Need to push the deadline by two days. Can we adjust?",
  "platform": "LinkedIn"
}
```

Sample response:
```json
{
  "original_message": "Need to push the deadline by two days. Can we adjust?",
  "tone_analysis": {
    "overall_tone": "Neutral",
    "rationale": "Heuristic analysis based on keywords and phrasing."
  },
  "vibes": [
    {
      "vibe": "Professional",
      "rewritten_text": "[Professional] Need to push the deadline by two days. Can we adjust?",
      "explanation": "Uses professional tone cues based on simple template guidance.",
      "use_cases": [
        "Use when you need a professional tone.",
        "Useful for quick edits when time is limited."
      ]
    },
    {
      "vibe": "Friendly",
      "rewritten_text": "[Friendly] Need to push the deadline by two days. Can we adjust?",
      "explanation": "Uses friendly tone cues based on simple template guidance.",
      "use_cases": [
        "Use when you need a friendly tone.",
        "Useful for quick edits when time is limited."
      ]
    },
    {
      "vibe": "Persuasive",
      "rewritten_text": "[Persuasive] Need to push the deadline by two days. Can we adjust?",
      "explanation": "Uses persuasive tone cues based on simple template guidance.",
      "use_cases": [
        "Use when you need a persuasive tone.",
        "Useful for quick edits when time is limited."
      ]
    },
    {
      "vibe": "Concise",
      "rewritten_text": "[Concise] Need to push the deadline by two days. Can we adjust?",
      "explanation": "Uses concise tone cues based on simple template guidance.",
      "use_cases": [
        "Use when you need a concise tone.",
        "Useful for quick edits when time is limited."
      ]
    },
    {
      "vibe": "Empathetic",
      "rewritten_text": "[Empathetic] Need to push the deadline by two days. Can we adjust?",
      "explanation": "Uses empathetic tone cues based on simple template guidance.",
      "use_cases": [
        "Use when you need an empathetic tone.",
        "Useful for quick edits when time is limited."
      ]
    }
  ],
  "platform_tips": {
    "platform": "linkedin",
    "tips": "Stay professional, avoid slang, include a clear ask, and keep paragraphs short."
  }
}
```

### Development notes
- The server uses async OpenAI client when available; without a key it falls back to local heuristics.
- Keep requests short to minimize token usage. Messages are truncated if very long.


