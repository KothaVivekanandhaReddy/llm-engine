from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import json

import faiss
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = Path("experiments/faiss.index")
METADATA_PATH = Path("experiments/faiss_metadata.json")


model = None
tokenizer = None
embedder = None
index = None
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, embedder, index, metadata

    print("Loading Qwen...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    print("Loading retrieval model...")

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Model and retriever ready.")

    yield

    print("Shutting down.")


app = FastAPI(
    title="Recollia LLM Engine",
    version="0.1.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    question: str
    max_new_tokens: int = 100


def generate_with_context(question, context, max_new_tokens):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical education assistant. "
                "Answer using only the provided context. "
                "If the context is insufficient, say so. "
                "Keep the answer concise and educational."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    start = perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    latency = perf_counter() - start

    input_tokens = inputs["input_ids"].shape[-1]
    output_tokens = outputs.shape[-1] - input_tokens

    answer = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    ).strip()

    return answer, latency, output_tokens


def retrieve(question, k=3):
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        doc = metadata[idx]

        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(float(score), 4),
            }
        )

    return results


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "retrieval": "faiss",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    answer, latency, output_tokens = generate_with_context(
        request.question,
        "",
        request.max_new_tokens,
    )

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3),
        "output_tokens": output_tokens,
        "tokens_per_second": round(
            output_tokens / latency,
            2,
        ),
    }


@app.post("/rag")
def rag(request: GenerateRequest):
    total_start = perf_counter()

    results = retrieve(
        request.question,
        k=3,
    )

    context = "\n\n".join(
        f"[{result['title']}]\n{result['text']}"
        for result in results
    )

    answer, generation_latency, output_tokens = generate_with_context(
        request.question,
        context,
        request.max_new_tokens,
    )

    total_latency = perf_counter() - total_start

    return {
        "answer": answer,
        "sources": [
            {
                "id": result["id"],
                "title": result["title"],
                "score": result["score"],
            }
            for result in results
        ],
        "generation_latency_seconds": round(
            generation_latency,
            3,
        ),
        "total_latency_seconds": round(
            total_latency,
            3,
        ),
        "output_tokens": output_tokens,
        "tokens_per_second": round(
            output_tokens / generation_latency,
            2,
        ),
    }