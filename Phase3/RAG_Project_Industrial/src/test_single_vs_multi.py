import time

from retriever import retrieve
from .multi_query_retriever import multi_query_retrieve


queries = [
    "docker port expose?",
    "docker volume",
    "docker network?",
    "how run docker container",
    "fastapi route?",
    "fastapi auth",
    "what is uvicorn used for?",
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    # ---------------------------------
    # SINGLE QUERY
    # ---------------------------------

    start = time.perf_counter()

    single_results = retrieve(
        query,
        top_k=10,
        score_threshold=0.30,
    )

    single_time = time.perf_counter() - start

    # ---------------------------------
    # MULTI QUERY
    # ---------------------------------

    start = time.perf_counter()

    multi_result = multi_query_retrieve(
        query,
        num_queries=3,
        top_k_per_query=10,
        final_k=10,
        score_threshold=0.30,
    )

    multi_time = time.perf_counter() - start

    # ---------------------------------
    # RESULTS
    # ---------------------------------

    print("\nSingle-query results:")
    print(
        [
            round(point.score, 4)
            for point in single_results[:5]
        ]
    )

    print("\nMulti-query results:")
    print(
        [
            round(point.score, 4)
            for point in multi_result["results"][:5]
        ]
    )

    print("\nGenerated queries:")

    for generated_query in multi_result["generated_queries"]:
        print(" -", generated_query)

    print("\nLatency:")
    print(f"Single query: {single_time:.4f}s")
    print(f"Multi query : {multi_time:.4f}s")