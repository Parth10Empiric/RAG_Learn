from pathlib import Path

import chromadb


CHROMA_PATH = Path("vector_store")

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="developer_knowledge"
)


def add_chunks(chunks: list[str], embeddings, source: str):
    ids = [
        f"{source}- chunk - {i}"
        for i in range(len(chunks))
    ]
    
    metadatas = [
        {
            "source": source,
            "chunk_index": i
        }
        for i in range (len(chunks))
    ]
    
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    

    