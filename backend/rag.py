from sentence_transformers import SentenceTransformer
from document_loader import load_documents
from chroma_manager import collection

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

documents = load_documents(
    "data/sample-docs"
)

# Clear existing documents
try:
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    print("Existing documents removed")

except Exception as e:
    print(f"Cleanup skipped: {e}")

# Ingest documents
for idx, doc in enumerate(documents):

    embedding = model.encode(
        doc["content"]
    ).tolist()

    collection.add(
        ids=[str(idx)],
        documents=[doc["content"]],
        metadatas=[
            {
                "source": doc["source"]
            }
        ],
        embeddings=[embedding]
    )

print(f"Added {len(documents)} documents")
