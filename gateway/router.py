from gateway.models import GenerateRequest, GenerateResponse
from gateway.providers import LocalQwenProvider
from gateway.mock_provider import FailingProvider


class ModelRouter:

    def __init__(self):
        self.providers = {
            "local-qwen": LocalQwenProvider(),
            "mock-failure": FailingProvider(),
        }

    def generate(self, request: GenerateRequest) -> GenerateResponse:

        if request.model not in self.providers:
            raise ValueError(
                f"Unknown model: {request.model}"
            )

        try:
            return self.providers[request.model].generate(request)

        except Exception as error:

            print(f"Provider failed: {error}")

            fallback_model = "local-qwen"

            if request.model != fallback_model:

                print(
                    f"Falling back to: {fallback_model}"
                )

                fallback_request = GenerateRequest(
                    prompt=request.prompt,
                    model=fallback_model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

                return self.providers[fallback_model].generate(
                    fallback_request
                )

            raise