from contextlib import asynccontextmanager
from time import perf_counter

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from gateway.models import GenerateRequest as GatewayGenerateRequest
from gateway.router import ModelRouter


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router

    print("Loading inference gateway...")

    router = ModelRouter(
        model_name=MODEL_NAME,
    )

    print("Inference gateway ready.")

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


class GatewayRequest(BaseModel):
    prompt: str
    model: str
    max_tokens: int = 100
    temperature: float = 0.0


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "retrieval": "faiss",
    }


def call_gateway(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
):

    request = GatewayGenerateRequest(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    start = perf_counter()

    response = router.generate(request)

    total_latency = (
        perf_counter() - start
    )

    return {
        "answer": response.text,
        "model": response.model,
        "provider": response.provider,
        "latency_seconds": round(
            response.latency_seconds,
            3,
        ),
        "total_latency_seconds": round(
            total_latency,
            3,
        ),
        "output_tokens": response.output_tokens,
        "metadata": response.metadata,
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    return call_gateway(
        prompt=request.question,
        model="local-qwen",
        max_tokens=request.max_new_tokens,
    )


@app.post("/rag")
def rag(request: GenerateRequest):

    return call_gateway(
        prompt=request.question,
        model="local-rag",
        max_tokens=request.max_new_tokens,
    )


@app.post("/gateway/generate")
def gateway_generate(
    request: GatewayRequest,
):

    return call_gateway(
        prompt=request.prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )