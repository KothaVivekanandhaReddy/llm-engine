import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:

    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model=EMBEDDING_MODEL,
    ):
        print("Loading retriever...")

        self.model = SentenceTransformer(
            embedding_model
        )

        self.index = faiss.read_index(
            str(index_path)
        )

        with Path(metadata_path).open(
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        print(
            f"Retriever ready. "
            f"Documents: {len(self.metadata)}"
        )

    def search(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if idx < 0:
                continue

            doc = self.metadata[idx]

            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )

        return results


def main():

    query = input("\nQuestion: ").strip()

    retriever = Retriever()

    results = retriever.search(
        query,
        top_k=3,
    )

    print("\nTop results:")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{rank}. "
            f"{result['title']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(result["text"])


if __name__ == "__main__":
    main()