import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Repo root (RAG_Learn) on sys.path so the Phase2 package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Phase2.rag_developer_assistant.store import client, COLLECTION_NAME

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_m3": "BAAI/bge-m3",
}

K_VALUES = [1, 3, 5, 10]


def make_key(source, chunk_index):
    return (
        source,
        int(chunk_index),
    )


def load_all_points():

    points = []

    offset = None

    while True:

        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(batch)

        if next_offset is None:
            break

        offset = next_offset

    return points

def calculate_metrics(
    ranked_keys,
    gold_keys,
):

    results = {}

    for k in K_VALUES:

        top_k = ranked_keys[:k]

        relevant = sum(
            key in gold_keys
            for key in top_k
        )

        results[f"hit@{k}"] = int(
            relevant > 0
        )

        results[f"precision@{k}"] = (
            relevant / k
        )

        results[f"recall@{k}"] = (
            relevant / len(gold_keys)
            if gold_keys
            else 0.0
        )

    reciprocal_rank = 0.0

    for rank, key in enumerate(
        ranked_keys,
        start=1,
    ):

        if key in gold_keys:

            reciprocal_rank = 1.0 / rank
            break

    results["mrr"] = reciprocal_rank

    return results

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


points = load_all_points()

documents = []

for point in points:

    payload = point.payload

    documents.append(
        {
            "text": payload["text"],
            "source": payload["source"],
            "chunk_index": payload["chunk_index"],
        }
    )

texts = [
    item["text"]
    for item in documents
]


print(
    f"Loaded {len(texts)} chunks from Qdrant."
)

for model_key, model_name in MODELS.items():

    print("\n" + "=" * 80)

    print(
        f"MODEL: {model_key}"
    )

    print(
        f"NAME: {model_name}"
    )

    print("=" * 80)


    model = SentenceTransformer(
        model_name
    )


    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


    aggregate = {
        "hit@1": [],
        "hit@3": [],
        "hit@5": [],
        "hit@10": [],
        "precision@1": [],
        "precision@3": [],
        "precision@5": [],
        "precision@10": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
    }


    for question_id, labels in qrels.items():

        question = questions[
            question_id
        ]["question"]


        # BGE-M3 does not require the
        # E5 query/passage prefixes.
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        )


        scores = (
            query_embedding
            @ document_embeddings.T
        )


        ranked_indices = np.argsort(
            scores
        )[::-1]


        ranked_keys = [

            make_key(
                documents[index]["source"],
                documents[index]["chunk_index"],
            )

            for index in ranked_indices
        ]


        gold_keys = {
            make_key(
                item["source"],
                item["chunk_index"],
            )

            for item in labels["relevant"]
        }


        metrics = calculate_metrics(
            ranked_keys,
            gold_keys,
        )


        for name, value in metrics.items():
            aggregate[name].append(
                value
            )


    print("\nAverage metrics:")

    for name, values in aggregate.items():

        average = (
            sum(values)
            / len(values)
        )

        print(
            f"{name}: {average:.4f}"
        )
        
        
