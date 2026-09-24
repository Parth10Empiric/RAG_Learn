from src.rag_pipeline import retrieve_with_rewrite
from retriever import retrieve


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for i, point in enumerate(results, start=1):
        payload = point.payload or {}

        print(f"\nRank: {i}")
        print(f"Score: {point.score:.4f}")
        print(f"Document: {payload.get('document_id')}")
        print(f"Source: {payload.get('source')}")
        print(f"Text: {payload.get('text', '')[:180]}...")


query = "docker port expose?"


# --------------------------------------------------
# ORIGINAL RETRIEVAL
# --------------------------------------------------

original_results = retrieve(
    query,
    document_id="docker",
    file_type="pdf",
)

print_results(
    "ORIGINAL QUERY",
    original_results,
)


# --------------------------------------------------
# REWRITTEN RETRIEVAL
# --------------------------------------------------

rewritten_query, rewritten_results = retrieve_with_rewrite(
    query,
    document_id="docker",
    file_type="pdf",
)

print("\n")
print("=" * 70)
print("QUERY REWRITING")
print("=" * 70)

print("Original query  :", query)
print("Rewritten query :", rewritten_query)

print_results(
    "REWRITTEN QUERY",
    rewritten_results,
)