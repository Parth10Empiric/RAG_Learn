import requests
from config import OLLAMA_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = OLLAMA_MODEL

def rewrite_query(query: str) -> str:
    prompt = f"""
You are a search query rewriting system for a technical documentation RAG system.

Rewrite the user's query into one clear, precise search query.

Rules:
- Preserve the original intent exactly.
- Do not answer the question.
- Do not add assumptions or unrelated information.
- Preserve technical terms.
- Expand vague wording when useful.
- Return only the rewritten query.
- Do not use quotation marks.
- Do not use markdown.
- Do not explain your reasoning.

User query:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    response.raise_for_status()

    rewritten = response.json()["response"].strip()

    rewritten = rewritten.strip("\"'")

    return rewritten