import json
from collections import defaultdict

from inspect_candidates import retrieve

K_VALUES = [1, 3, 5, 10]

def make_key(source, chunk_index):
    return (
        source,
        int(chunk_index),
    )
    
def evaluate_query(
    question,
    gold_items,
):
    results = retrieve(
        query=question,
        top_k=max(K_VALUES),
    )
    
    retrieved_keys = [
        make_key(
            point.payload["source"],
            point.payload["chunk_index"],
        )
        for point in results
    ]
    
    gold_keys = {
        make_key(
            item["source"],
            item["chunk_index"],
        )
        for item in gold_items
    }

    metrics = {}
    
    for k in K_VALUES:
        top_k = retrieved_keys[:k]
        
        relevant_count = sum(
            key in gold_keys
            for key in top_k
        )
        
        hit = (
            1
            if relevant_count > 0
            else 0
        )

        precision = (
            relevant_count / k
            if k > 0
            else 0
        )

        recall = (
            relevant_count / len(gold_keys)
            if gold_keys
            else 0
        )

        metrics[k] = {
            "hit": hit,
            "precision": precision,
            "recall": recall,
        }

        reciprocal_rank = 0.0

    for rank, key in enumerate(
        retrieved_keys,
        start=1,
    ):

        if key in gold_keys:

            reciprocal_rank = 1.0 / rank
            break

    return metrics, reciprocal_rank


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
    
    
aggregate = defaultdict(list)


for question_id, labels in qrels.items():

    question = questions[question_id][
        "question"
    ]

    metrics, reciprocal_rank = evaluate_query(
        question=question,
        gold_items=labels["relevant"],
    )

    print("\n" + "=" * 70)

    print(
        f"{question_id}: {question}"
    )

    for k, values in metrics.items():

        print(
            f"Hit@{k}: "
            f"{values['hit']}"
        )

        print(
            f"Precision@{k}: "
            f"{values['precision']:.3f}"
        )

        print(
            f"Recall@{k}: "
            f"{values['recall']:.3f}"
        )

        aggregate[
            f"hit@{k}"
        ].append(values["hit"])

        aggregate[
            f"precision@{k}"
        ].append(values["precision"])

        aggregate[
            f"recall@{k}"
        ].append(values["recall"])

    print(
        f"Reciprocal Rank: "
        f"{reciprocal_rank:.3f}"
    )

    aggregate[
        "mrr"
    ].append(reciprocal_rank)


print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)


for metric, values in aggregate.items():

    average = sum(values) / len(values)

    print(
        f"{metric}: {average:.4f}"
    )
    