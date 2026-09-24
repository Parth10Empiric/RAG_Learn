from .multi_query import generate_queries


queries = [
    "docker port expose?",
    "docker volume",
    "docker network?",
    "how run docker container",
    "fastapi route?",
    "fastapi auth",
    "what is uvicorn used for?",
]


for original_query in queries:

    generated_queries = generate_queries(
        original_query,
        num_queries=3,
    )

    print("\n" + "=" * 70)
    print("ORIGINAL QUERY")
    print("=" * 70)

    print(original_query)

    print("\nGENERATED QUERIES")
    print("-" * 70)

    for i, generated_query in enumerate(
        generated_queries,
        start=1,
    ):
        print(f"{i}. {generated_query}")