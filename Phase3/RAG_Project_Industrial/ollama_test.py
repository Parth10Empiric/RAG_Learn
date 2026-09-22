from ollama import chat


response = chat(
    model="qwen2.5-coder:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain FastAPI in one sentence."
        }
    ]
)

print(response.message.content)