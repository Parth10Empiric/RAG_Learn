from .retriever import retrieve


query = "How does FastAPI validate request data?"

results = retrieve(
    query=query,
    top_k=3
)

print("\nQuery:")
print(query)

print("\nResults:")

    
for rank, point in enumerate(results, start=1):

    print(f"\n--- Result {rank} ---")

    print("Score:", point.score)

    print("Source:", point.payload["source"])

    print("Chunk:", point.payload["chunk_index"])

    print("Text:")
    print(point.payload["text"])
    
