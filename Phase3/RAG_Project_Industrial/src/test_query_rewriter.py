from .query_rewriter import rewrite_query

queries = [
    "docker port expose?",
    "how run docker container",
    "fastapi route?",
    "fastapi auth",
    "what is uvicorn used for?",
]


for query in queries:
    rewritten = rewrite_query(query)

    print("\n" + "=" * 60)
    print("Original : ", query)
    print("Rewritten:", rewritten)
    
    
    