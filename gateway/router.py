from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from gateway.models import GenerateRequest, GenerateResponse
from gateway.providers import LocalQwenProvider
from gateway.rag_provider import LocalRAGProvider
from gateway.mock_provider import FailingProvider


class ModelRouter:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer=None,
        model=None,
    ):

        if tokenizer is None or model is None:

           print(f"Loading shared model: {model_name}")

           tokenizer = AutoTokenizer.from_pretrained(
                model_name
           )

           model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
           )

           model.eval()

           print("Shared model ready.")

        self.providers = {
            "local-qwen": LocalQwenProvider(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
            ),

            "local-rag": LocalRAGProvider(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
            ),

            "mock-failure": FailingProvider(),
        }

    def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:

        if request.model not in self.providers:
            raise ValueError(
                f"Unknown model: {request.model}"
            )

        try:

            return self.providers[
                request.model
            ].generate(request)

        except Exception as error:

            print(
                f"Provider failed: {error}"
            )

            fallback_model = "local-qwen"

            if request.model != fallback_model:

                print(
                    f"Falling back to: "
                    f"{fallback_model}"
                )

                fallback_request = GenerateRequest(
                    prompt=request.prompt,
                    model=fallback_model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

                return self.providers[
                    fallback_model
                ].generate(
                    fallback_request
                )

            raise