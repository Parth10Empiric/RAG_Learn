import json

from ..retriever import retrieve


TOP_K = 20


with open(
    "RAG_Project_Industrial/evaluation/retrieval_questions.json",
    "r",
    encoding="utf-8",
) as file:
    questions = json.load(file)


for item in questions:

    print("\n" + "=" * 80)

    print(
        f"{item['id']}: "
        f"{item['question']}"
    )

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

        print(
            f"Rank {rank:2d} | "
            f"Score {point.score:.4f} | "
            f"{point.payload['source']} | "
            f"chunk {point.payload['chunk_index']}"
        )