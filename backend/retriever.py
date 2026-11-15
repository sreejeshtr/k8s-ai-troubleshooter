from chroma_manager import collection


def retrieve_context(query, n_results=3):

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n".join(documents)

    ranking = []

    for idx, item in enumerate(metadata):

        confidence = round(
            (1 / (1 + distances[idx])) * 100,
            1
        )

        ranking.append(
            {
                "rank": idx + 1,
                "source": item["source"],
                "distance": distances[idx],
                "confidence": confidence
            }
        )

    return context, ranking
