from .config import TOP_K, SCORE_THRESHOLD
from .embedder import embed_query
from .store import client, COLLECTION_NAME


def retrieve(query: str, top_k: int = TOP_K, score_threshold: float | None = SCORE_THRESHOLD):
    query_embedding = embed_query(query)
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return response.points
