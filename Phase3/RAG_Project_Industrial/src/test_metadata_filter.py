from ..retriever import retrieve

def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    if not results:
        print("No results found.")
        return
    
    for i, point in enumerate(results, start=1):
        payload = point.payload or {}
        
        print(f"\nRank: {i}")
        print(f"Score: {point.score:.4f}")
        print(f"Document ID: {payload.get('document_id')}")
        print(f"File Type: {payload.get('file_type')}")
        print(f"Chunk Type: {payload.get('chunk_type')}")
        print(f"Source: {payload.get('source')}")
        print(f"Text: {payload.get('text', '')[:150]}...")

query = "How do I create and run a Docker container?"




# 1. No filter
results = retrieve(query)

print_results(
    "1. UNFILTERED RETRIEVAL",
    results
)


# 2. Filter by document
results = retrieve(
    query,
    document_id="docker"
)

print_results(
    "2. FILTER: document_id=docker",
    results
)


# 3. Filter by file type
results = retrieve(
    query,
    file_type="pdf"
)

print_results(
    "3. FILTER: file_type=pdf",
    results
)


# 4. Combine filters
results = retrieve(
    query,
    document_id="docker",
    file_type="pdf"
)

print_results(
    "4. COMBINED FILTER: document_id=docker AND file_type=pdf",
    results
)


# 5. Wrong filter
results = retrieve(
    query,
    document_id="fastapi"
)

print_results(
    "5. WRONG FILTER: document_id=fastapi",
    results
)