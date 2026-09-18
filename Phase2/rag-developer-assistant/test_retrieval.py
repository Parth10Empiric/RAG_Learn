from retriever import retrieve


query = "How does FastAPI validate request data?"

results = retrieve(
    query=query,
    top_k=3
)

print("\nQuery:")
print(query)

print("\nResults:")

for i, document in enumerate(results["documents"][0], start=1):

    print(f"\n--- Result {i} ---")
    print(document)
    
for i in range(len(results["documents"][0])):

    print(f"\n--- Result {i + 1} ---")

    print("Text:")
    print(results["documents"][0][i])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nDistance:")
    print(results["distances"][0][i])
    
