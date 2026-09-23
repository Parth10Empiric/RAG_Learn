from ..retriever import retrieve


TEST_CASES = [
    {
        "query": "How do I create and run a Docker container?",
        "expected_document": "docker",
    },
    {
        "query": "How do I define a FastAPI route?",
        "expected_document": "fastapi",
    },
    {
        "query": "How do I configure FastAPI?",
        "expected_document": "fastapi",
    },
    {
        "query": "How do Docker containers expose ports?",
        "expected_document": "docker",
    },
]


def contains_document(results, expected_document):
    for point in results:
        payload = point.payload or {}

        if payload.get("document_id") == expected_document:
            return True

    return False


def evaluate(name, use_filter=False):
    correct = 0

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for case in TEST_CASES:

        if use_filter:
            results = retrieve(
                case["query"],
                document_id=case["expected_document"],
            )
        else:
            results = retrieve(
                case["query"]
            )

        found = contains_document(
            results,
            case["expected_document"]
        )

        if found:
            correct += 1

        print(
            f"\nQuery: {case['query']}"
        )
        print(
            f"Expected document: {case['expected_document']}"
        )
        print(
            f"Expected document found: {found}"
        )

    accuracy = correct / len(TEST_CASES)

    print("\n----------------------------------------")
    print(f"Correct: {correct}/{len(TEST_CASES)}")
    print(f"Document hit rate: {accuracy:.4f}")


evaluate(
    "UNFILTERED RETRIEVAL",
    use_filter=False,
)

evaluate(
    "DOCUMENT-FILTERED RETRIEVAL",
    use_filter=True,
)