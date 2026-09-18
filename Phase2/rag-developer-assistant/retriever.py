from embedder import embed_query
from store import client, COLLECTION_NAME


def retrieve(query: str, top_k:int = 3):
    query_embedding = embed_query(query)
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )

    return response.points
