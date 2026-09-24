import json

from retriever import retrieve
from src.test_multi_query_retriever import multi_query_retrieve

THRESHOLDS = [
    None,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60
]


TOP_K = 10


def make_key(source, chunk_index):
    return (
        source,
        int(chunk_index),
    )


def evaluate_question(
    question,
    gold_items,
    threshold,
):
    results = retrieve(
        query=question,
        top_k=TOP_K,
        score_threshold=threshold,
    )

    multi_result = multi_query_retrieve(
        query=question,
        num_queries=3,
        top_k_per_query=10,
        final_k=10,
        score_threshold=0.30,
    )

    multi_results = multi_result["results"]

    retrieved_keys = [
        make_key(
            point.payload["source"],
            point.payload["chunk_index"],
        )
        for point in multi_results
    ]

    gold_keys = {
        make_key(
            item["source"],
            item["chunk_index"],
        )
        for item in gold_items
    }

    relevant_count = sum(
        key in gold_keys
        for key in retrieved_keys
    )

    returned_count = len(
        retrieved_keys
    )

    hit = int(
        relevant_count > 0
    )

    precision = (
        relevant_count / returned_count
        if returned_count > 0
        else 0.0
    )

    recall = (
        relevant_count / len(gold_keys)
        if gold_keys
        else 0.0
    )

    return {
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "returned": returned_count,
    }


with open(
    "evaluation/retrieval_questions.json",
    "r",
    encoding="utf-8",
) as file:
    questions = {
        item["id"]: item
        for item in json.load(file)
    }


with open(
    "evaluation/qrels.json",
    "r",
    encoding="utf-8",
) as file:
    qrels = json.load(file)


for threshold in THRESHOLDS:

    results = {
        "hit": [],
        "precision": [],
        "recall": [],
        "returned": [],
    }

    for question_id, labels in qrels.items():

        question = questions[
            question_id
        ]["question"]

        values = evaluate_question(
            question=question,
            gold_items=labels["relevant"],
            threshold=threshold,
        )

        for key, value in values.items():
            results[key].append(value)

    print("\n" + "=" * 70)

    label = (
        "NONE"
        if threshold is None
        else f"{threshold:.2f}"
    )

    print(
        f"THRESHOLD = {label}"
    )

    print("=" * 70)

    for key, values in results.items():

        average = (
            sum(values)
            / len(values)
        )

        print(
            f"{key}: {average:.4f}"
        )