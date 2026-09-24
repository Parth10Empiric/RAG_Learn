from src.query_rewriter import rewrite_query
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


# Original query
original_results = retrieve(query)

print_results(
    "ORIGINAL QUERY",
    original_results
)


# Rewrite
rewritten_query = rewrite_query(query)

print("\n")
print("=" * 70)
print("ORIGINAL QUERY :", query)
print("REWRITTEN QUERY:", rewritten_query)
print("=" * 70)


# Rewritten query
rewritten_results = retrieve(rewritten_query)

print_results(
    "REWRITTEN QUERY",
    rewritten_results
)