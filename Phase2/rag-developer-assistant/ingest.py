from loader import load_text_file
from chunker import chunk_text
from embedder import embed_documents
from store import add_chunks


SOURCE = "fastapi.txt"
PATH = "data/documents/fastapi.txt"


# 1. Load
text = load_text_file(PATH)

print(f"Loaded {SOURCE}")
print(f"Characters: {len(text)}")

# 2. chunk
chunks = chunk_text(text, chunk_size=200, overlap=40)

print(f"Created {len(chunks)} chunks")

# 3. Embed
embeddings = embed_documents(chunks)

print(f"Created embeddings: {embeddings.shape}")

# 4. store
add_chunks(chunks=chunks, embeddings=embeddings, source=SOURCE)

print("Stored chunks successfully.")