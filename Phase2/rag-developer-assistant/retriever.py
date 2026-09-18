from embedder import embed_query
from store import collection


def retrieve(query: str, top_k:int = 3):
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results
