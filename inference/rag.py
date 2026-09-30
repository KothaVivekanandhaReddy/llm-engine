import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever():
    print("Loading embedding model...")

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embedder, index, metadata


def retrieve(query, embedder, index, metadata, k=3):
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append(
            {
                "id": metadata[idx]["id"],
                "title": metadata[idx]["title"],
                "text": metadata[idx]["text"],
                "score": float(score),
            }
        )

    return results


def load_llm():
    print("Loading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    return model, tokenizer


def generate_answer(question, context, model, tokenizer):
    system_prompt = """You are a medical education assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer,
say that the provided context is insufficient.

Keep the answer concise and educational.
Do not invent medical facts."""


    user_prompt = f"""Context:

{context}

Question:
{question}

Answer:"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return answer


def main():
    question = input("\nQuestion: ").strip()

    embedder, index, metadata = load_retriever()

    results = retrieve(
        question,
        embedder,
        index,
        metadata,
        k=3,
    )

    context = "\n\n".join(
        f"[{r['title']}]\n{r['text']}"
        for r in results
    )

    model, tokenizer = load_llm()

    answer = generate_answer(
        question,
        context,
        model,
        tokenizer,
    )

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        print(
            f"\n{i}. {result['title']} "
            f"(score={result['score']:.4f})"
        )

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()