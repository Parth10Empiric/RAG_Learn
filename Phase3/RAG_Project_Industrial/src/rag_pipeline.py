from src.query_rewriter import rewrite_query
from retriever import retrieve

import time

def retrieve_with_rewrite(
    query: str,
    top_k: int = 10,
    score_threshold: float | None = 0.30,
    document_id: str | None = None,
    file_type: str | None = None,
):
    start = time.perf_counter()

    rewritten_query = rewrite_query(query)

    rewrite_time = time.perf_counter() - start
    print("="*20)
    print("Rewrite_Time: ", rewrite_time)
    print("="*20)
    retrieval_start = time.perf_counter()

    retrieval_time = time.perf_counter() - retrieval_start
    
    print("="*20)
    print("retrieval_time: ", retrieval_time)
    print("="*20)
    
    results = retrieve(
        rewritten_query,
        top_k=top_k,
        score_threshold=score_threshold,
        document_id=document_id,
        file_type=file_type,
    )

    return rewritten_query, results
