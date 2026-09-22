from .retriever import retrieve
from .generator import generate_answer


def build_context(results):
    sections = []

    for i, point in enumerate(results, start=1):
        source = point.payload["source"]
        chunk_index = point.payload["chunk_index"]
        text = point.payload["text"]

        sections.append(
            f"[Source {i}]\n"
            f"File: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Text: {text}"
        )

    return "\n\n".join(sections)


query = input("\nAsk a question: ")


results = retrieve(query=query)


context = build_context(results)


answer = generate_answer(
    query=query,
    context=context,
)


print("\n================ ANSWER ================")
print(answer)


print("\n================ SOURCES ===============")

for i, point in enumerate(results, start=1):
    print(
        f"{i}. "
        f"{point.payload['source']} "
        f"(chunk {point.payload['chunk_index']}) "
        f"[score={point.score:.4f}]"
    )