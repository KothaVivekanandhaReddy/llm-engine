import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/raw/medical_docs.json")
INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded documents: {len(documents)}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [doc["text"] for doc in documents]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
        }
        for doc in documents
    ]

    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Ingestion complete.")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {embeddings.shape[1]}")
    print(f"Index: {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()