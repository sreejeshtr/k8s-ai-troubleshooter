from pathlib import Path


def load_documents(folder):

    documents = []

    for file in Path(folder).glob("*.txt"):

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        documents.append(
            {
                "source": file.name,
                "content": content
            }
        )

    return documents
