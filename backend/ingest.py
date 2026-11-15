import uuid

from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from chroma_manager import collection


def ingest_documents(folder):

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    docs = load_documents(folder)

    count = 0

    for doc in docs:

        embedding = model.encode(
            doc["content"]
        ).tolist()

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc["content"]],
            metadatas=[
                {
                    "source": doc["source"]
                }
            ],
            embeddings=[embedding]
        )

        count += 1

    return count
