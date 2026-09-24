from config import TOP_K, SCORE_THRESHOLD
from embedder import embed_query
from store import client, COLLECTION_NAME
from qdrant_client import models

def retrieve(
    query: str, 
    top_k: int = TOP_K, 
    score_threshold: float | None = SCORE_THRESHOLD,
    document_id: str | None = None,
    file_type: str | None = None
):
    
    must_conditions = []

    if document_id:
        must_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )

    if file_type:
        must_conditions.append(
            models.FieldCondition(
                key="file_type",
                match=models.MatchValue(value=file_type),
            )
        )

    query_filter = (
        models.Filter(must=must_conditions)
        if must_conditions
        else None
    )

    query_embedding = embed_query(query)
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return response.points
