import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    query = input("\nQuestion: ").strip()

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        3,
    )

    print("\nTop results:")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        doc = metadata[idx]

        print(f"\n{rank}. {doc['title']}")
        print(f"Score: {score:.4f}")
        print(doc["text"])


if __name__ == "__main__":
    main()