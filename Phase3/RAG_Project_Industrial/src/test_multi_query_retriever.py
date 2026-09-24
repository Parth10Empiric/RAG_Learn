from .multi_query_retriever import multi_query_retrieve


def print_results(result):

    print("\n" + "=" * 80)
    print("ORIGINAL QUERY")
    print("=" * 80)

    print(result["original_query"])

    print("\nGENERATED QUERIES")
    print("-" * 80)

    for query in result["generated_queries"]:
        print(query)

    print("\nFINAL RESULTS")
    print("-" * 80)

    for rank, point in enumerate(
        result["results"],
        start=1,
    ):

        payload = point.payload or {}

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {point.score:.4f}"
        )

        print(
            f"Document: {payload.get('document_id')}"
        )

        print(
            f"Source: {payload.get('source')}"
        )

        print(
            f"Text: {payload.get('text', '')[:200]}"
        )


query = "docker port expose?"

result = multi_query_retrieve(
    query=query,
    num_queries=3,
    top_k_per_query=10,
    final_k=10,
    score_threshold=0.30,
)

print_results(result)