from chroma_manager import collection

results = collection.query(
    query_texts=[
        "Why am I getting ImagePullBackOff?"
    ],
    n_results=2
)

print(results)
