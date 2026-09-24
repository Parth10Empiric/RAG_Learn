from .multi_query import generate_queries
from retriever import retrieve


def multi_query_retrieve(
    query: str,
    num_queries: int = 3,
    top_k_per_query: int = 10,
    final_k: int = 10,
    score_threshold: float | None = 0.30,
    document_id: str | None = None,
    file_type: str | None = None,
):

    generated_queries = generate_queries(
        query,
        num_queries=num_queries,
    )

    all_queries = [
        query,
        *generated_queries,
    ]

    merged_results = {}

    query_results = {}

    for search_query in all_queries:

        results = retrieve(
            search_query,
            top_k=top_k_per_query,
            score_threshold=score_threshold,
            document_id=document_id,
            file_type=file_type,
        )

        query_results[search_query] = results

        for rank, point in enumerate(results, start=1):

            point_id = str(point.id)

            if point_id not in merged_results:

                merged_results[point_id] = {
                    "point": point,
                    "best_score": point.score,
                    "occurrences": 1,
                    "best_rank": rank,
                    "matched_queries": [search_query],
                }

            else:

                entry = merged_results[point_id]

                entry["occurrences"] += 1
                entry["matched_queries"].append(search_query)

                if point.score > entry["best_score"]:

                    entry["best_score"] = point.score
                    entry["best_rank"] = rank
                    entry["point"] = point

    # Sort by best semantic score.
    ranked_results = sorted(
        merged_results.values(),
        key=lambda item: (
            item["best_score"],
            item["occurrences"],
        ),
        reverse=True,
    )

    final_points = [
        item["point"]
        for item in ranked_results[:final_k]
    ]

    return {
        "original_query": query,
        "generated_queries": generated_queries,
        "all_queries": all_queries,
        "query_results": query_results,
        "merged_results": ranked_results,
        "results": final_points,
    }