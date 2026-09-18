import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer (
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Python is a programming language.",
    "Python is commonly used for backend development.",
    "FastAPI is a Python framework for APIs.",
    "The weather is very hot today.",
]

embeddings = model.encode(texts)


similarity_matrix = cosine_similarity(embeddings)

print("Similarity matrix:")
print(similarity_matrix)
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = similarity_matrix[i][j]

        print("\nSentence A:")
        print(texts[i])

        print("Sentence B:")
        print(texts[j])

        print(f"Similarity: {score:.4f}")