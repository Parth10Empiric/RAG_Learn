# from pathlib import Path

# import chromadb


# CHROMA_PATH = Path("vector_store")

# client = chromadb.PersistentClient(
#     path=str(CHROMA_PATH)
# )

# collection = client.get_or_create_collection(
#     name="developer_knowledge"
# )


# def add_chunks(chunks: list[str], embeddings, source: str):
#     ids = [
#         f"{source}- chunk - {i}"
#         for i in range(len(chunks))
#     ]
    
#     metadatas = [
#         {
#             "source": source,
#             "chunk_index": i
#         }
#         for i in range (len(chunks))
#     ]
    
#     collection.upsert(
#         ids=ids,
#         documents=chunks,
#         embeddings=embeddings.tolist(),
#         metadatas=metadatas
#     )
    


import uuid

from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "developer_knowledge"
VECTOR_SIZE = 384

client = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

def add_chunks(
    chunks: list[str],
    embeddings,
    source: str,
):
    ensure_collection()

    points = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{source}:{i}",
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": source,
                    "chunk_index": i,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

