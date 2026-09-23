from pathlib import Path

from .loader import load_text_file, load_pdf_file
from .chunker import chunk_text
from .embedder import embed_documents
from .store import add_chunks


# Documents live inside the package, so resolve them from this file
# rather than from the current working directory.
DATA_DIR = Path(__file__).resolve().parent / "data" / "documents"

SOURCE = "fastapi.txt"
# SOURCE = "docker.pdf"
PATH = DATA_DIR / SOURCE

# 1. Load
text = load_text_file(PATH)
# text = load_pdf_file(PATH)

print(f"Loaded {SOURCE}")
print(f"Characters: {len(text)}")

# 2. chunk
chunks = chunk_text(text, chunk_size=500, overlap=75)

print(f"Created {len(chunks)} chunks")

# 3. Embed
embeddings = embed_documents(chunks)

print(f"Created embeddings: {embeddings.shape}")

# 4. store
add_chunks(chunks=chunks, embeddings=embeddings, source=SOURCE)

print("Stored chunks successfully.")