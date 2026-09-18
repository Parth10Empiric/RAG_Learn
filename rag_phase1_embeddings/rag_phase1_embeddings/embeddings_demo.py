from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Python is a programming language.",
    "FastAPI is a framework for building APIs.",
    "Docker packages applications into containers.",
]

embeddings = model.encode(texts)

print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)

for text, embedding in zip(texts, embeddings):
    print("\nText:")
    print(text)

    print("First 10 embedding values:")
    print(embedding[:10])