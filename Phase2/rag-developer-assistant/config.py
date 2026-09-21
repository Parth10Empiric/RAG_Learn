import os

from dotenv import load_dotenv


load_dotenv()


QDRANT_API_KEY = os.getenv("db_api_key")
QDRANT_ENDPOINT = os.getenv("CLUSTER_ENDPOINT")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)


if not QDRANT_API_KEY:
    raise RuntimeError(
        "Missing db_api_key in .env"
    )


if not QDRANT_ENDPOINT:
    raise RuntimeError(
        "Missing CLUSTER_ENDPOINT in .env"
    )