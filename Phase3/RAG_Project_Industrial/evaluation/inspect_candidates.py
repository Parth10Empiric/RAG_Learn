import json
import sys
from pathlib import Path

from ..retriever import retrieve


TOP_K = 10


with open("RAG_Project_Industrial/evaluation/retrieval_questions.json","r", encoding="utf-8",) as file :
    questions = json.load(file)
    
    
for item in questions:
    print("\n" + "=" * 80)

    print(f"ID: {item['id']}")
    print(f"Question: {item['question']}")

    print("=" * 80)

    results = retrieve(
        query=item["question"],
        top_k=TOP_K,
        score_threshold=None,
    )

    for rank, point in enumerate(
        results,
        start=1,
    ):

        payload = point.payload

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: {point.score:.4f}"
        )

        print(
            f"Source: {payload.get('source')}"
        )

        print(
            f"Chunk: {payload.get('chunk_index')}"
        )

        print(
            "Text:"
        )

        print(
            payload.get("text", "")[:1000]
        )