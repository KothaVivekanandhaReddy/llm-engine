import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


TEST_QUERIES = [
    {
        "question": "What does the kidney do?",
        "expected_id": "doc_002",
    },
    {
        "question": "What is oxygen saturation?",
        "expected_id": "doc_001",
    },
    {
        "question": "What controls blood glucose?",
        "expected_id": "doc_006",
    },
    {
        "question": "What is the role of DNA?",
        "expected_id": "doc_003",
    },
    {
        "question": "What is hypertension?",
        "expected_id": "doc_004",
    },
]


def recall_at_k(results, k):
    correct = 0

    for result in results:
        retrieved_ids = result["retrieved_ids"][:k]

        if result["expected_id"] in retrieved_ids:
            correct += 1

    return correct / len(results)


def main():
    print("Loading index...")

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    results = []

    for test in TEST_QUERIES:
        query_embedding = model.encode(
            [test["question"]],
            normalize_embeddings=True,
        )

        scores, indices = index.search(
            query_embedding,
            3,
        )

        retrieved_ids = [
            metadata[idx]["id"]
            for idx in indices[0]
        ]

        results.append(
            {
                "question": test["question"],
                "expected_id": test["expected_id"],
                "retrieved_ids": retrieved_ids,
                "scores": [
                    round(float(score), 4)
                    for score in scores[0]
                ],
            }
        )

    r1 = recall_at_k(results, 1)
    r3 = recall_at_k(results, 3)

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for result in results:
        print()
        print("Question:", result["question"])
        print("Expected:", result["expected_id"])
        print("Retrieved:", result["retrieved_ids"])
        print("Scores:", result["scores"])

    print()
    print("-" * 70)
    print(f"Recall@1: {r1:.2f}")
    print(f"Recall@3: {r3:.2f}")
    print("-" * 70)


if __name__ == "__main__":
    main()