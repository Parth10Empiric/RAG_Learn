import json

from ..retriever import retrieve


K_VALUES = [1, 3, 5, 10, 15, 20]

def make_key(source, chunk_index):
    return(source, int(chunk_index))


def evaluate_question(question, gold_items, top_k):
    
    results = retrieve(query=question, top_k=top_k, score_threshold=None)
    
    retrieved_keys = [
        make_key(point.payload["source"], point.payload["chunk_index"]) for point in results
    ]
    
    gold_keys = {
        make_key(
            item["source"],
            item["chunk_index"],
        )
        for item in gold_items
    }
    
    relevant_count = sum(key in gold_keys for key in retrieved_keys)
    
    hit = int(relevant_count > 0)
    
    precision = (
        relevant_count / top_k
    )
    
    recall = (
        relevant_count / len(gold_keys)
        if gold_keys
        else 0.0
    )
    
    reciprocal_rank = 0.0
    
    for rank, key in enumerate(
        retrieved_keys,
        start=1,
    ):
        if key in gold_keys:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
    }
    
with open(
    "RAG_Project_Industrial/evaluation/retrieval_questions.json",
    "r",
    encoding="utf-8",
) as file:
    questions = {
        item["id"]: item
        for item in json.load(file)
    }


with open(
    "RAG_Project_Industrial/evaluation/qrels.json",
    "r",
    encoding="utf-8",
) as file:
    qrels = json.load(file)

for k in K_VALUES:

    metrics = {
        "hit": [],
        "precision": [],
        "recall": [],
        "mrr": [],
    }

    for question_id, labels in qrels.items():

        question = questions[
            question_id
        ]["question"]

        result = evaluate_question(
            question=question,
            gold_items=labels["relevant"],
            top_k=k,
        )

        for metric_name, value in result.items():
            metrics[metric_name].append(value)

    print("\n" + "=" * 60)
    print(f"TOP-K = {k}")
    print("=" * 60)

    for metric_name, values in metrics.items():

        average = (
            sum(values) / len(values)
        )

        print(
            f"{metric_name}: {average:.4f}"
        )
