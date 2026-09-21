from ollama import chat


MODEL_NAME = "qwen2.5-coder:1.5b"


def generate_answer(
    query: str,
    context: str,
) -> str:

    system_prompt = """
You are a helpful developer knowledge assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not invent information.
2. If the context does not contain the answer, say:
   "I don't have enough information in the provided knowledge base."
3. Keep the answer clear and technically accurate.
"""

    user_prompt = f"""
Context:

{context}

Question:

{query}
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content
