import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gateway.models import GenerateRequest, GenerateResponse
from retrieval.retrieve import Retriever


class LocalRAGProvider:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer=None,
        model=None,
    ):
        print("Loading RAG provider...")

        self.model_name = model_name

        self.retriever = Retriever()

        if tokenizer is None or model is None:
            print(f"Loading {model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
            )

            self.model.eval()

        else:
            self.tokenizer = tokenizer
            self.model = model

        print("RAG provider ready.")

    def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:

        start = time.perf_counter()

        # -------------------------
        # 1. Retrieve context
        # -------------------------

        retrieval_start = time.perf_counter()

        retrieved = self.retriever.search(
            request.prompt,
            top_k=3,
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        context = "\n\n".join(
            (
                f"[{item['title']}]\n"
                f"{item['text']}"
            )
            for item in retrieved
        )

        # -------------------------
        # 2. Build prompt
        # -------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise medical education "
                    "assistant. Answer using the provided "
                    "context. Do not invent unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {request.prompt}"
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # -------------------------
        # 3. Generate
        # -------------------------

        generation_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=False,
            )

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        total_latency = (
            time.perf_counter()
            - start
        )

        # -------------------------
        # 4. Response
        # -------------------------

        return GenerateResponse(
            text=text,
            model=self.model_name,
            provider="local-rag",
            latency_seconds=total_latency,
            output_tokens=len(generated_tokens),
            metadata={
                "retrieval_latency_seconds":
                    retrieval_latency,

                "generation_latency_seconds":
                    generation_latency,

                "sources": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "score": item["score"],
                    }
                    for item in retrieved
                ],

                "device": (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                ),
            },
        )