from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]):
    return model.encode_document(
        texts,
        normalize_embeddings=True
    )
    
def embed_query(query: str):
    return model.encode_query(
        query,
        normalize_embeddings=True
    )
    
