import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    {
        "id": "doc1",
        "text": "FastAPI is a modern Python framework for building APIs."
    },
    {
        "id": "doc2",
        "text": "Redis is an in-memory data store often used for caching."
    },
    {
        "id": "doc3",
        "text": "Docker packages applications and dependencies into containers."
    },
    {
        "id": "doc4",
        "text": "Django is a Python framework for building web applications."
    },
    {
        "id": "doc5",
        "text": "MySQL is a relational database management system."
        
    }
]

print(documents[0]["text"])

for i in range(len(documents)):
    
    documents[i]["vector"] = model.encode_document(
        documents[i]["text"],
        normalize_embeddings=True
    )

query = "How can I create APIs with Python?"

query_embedding = model.encode_query(
    query,
    normalize_embeddings=True
)

doc_matrix = np.array([doc["vector"] for doc in documents])

scores = query_embedding @ doc_matrix.T

top_k = 3

top_indices = np.argsort(scores)[::-1][:top_k]

print("\nQuery:")
print(query)

print("\nTop results:")

for rank, index in enumerate(top_indices, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {scores[index]:.4f}")
    print(f"Document: {documents[index]['text']}")