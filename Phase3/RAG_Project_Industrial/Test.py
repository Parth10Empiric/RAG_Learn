from .loader import load_text_file
from .chunker import chunk_text
from .embedder import embed_documents


text = load_text_file("data/documents/fastapi.txt")

chunks = chunk_text(
    text,
    chunk_size=200,
    overlap=40
)

# for i, chunk in enumerate(chunks):
#     print(f"\n--- Chunk {i} ---")
#     print(chunk)
    


embeddings = embed_documents(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)