import time
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gateway.models import GenerateRequest, GenerateResponse


class BaseProvider(ABC):

    @abstractmethod
    def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        pass


class LocalQwenProvider(BaseProvider):

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer=None,
        model=None,
    ):
        self.model_name = model_name

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

            print("Local Qwen ready.")

        else:
            self.tokenizer = tokenizer
            self.model = model

    def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise medical education assistant. "
                    "Provide accurate educational explanations."
                ),
            },
            {
                "role": "user",
                "content": request.prompt,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0,
                temperature=(
                    request.temperature
                    if request.temperature > 0
                    else None
                ),
            )

        latency = time.perf_counter() - start

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return GenerateResponse(
            text=text,
            model=self.model_name,
            provider="local",
            latency_seconds=latency,
            output_tokens=len(generated_tokens),
            metadata={
               "device": (
                   "cuda"
                   if torch.cuda.is_available()
                   else "cpu"
               ),
               "tokens_per_second": (
                    len(generated_tokens) / latency
                    if latency > 0
                    else 0.0
                ),
},
)