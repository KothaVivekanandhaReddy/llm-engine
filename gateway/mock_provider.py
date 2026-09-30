from gateway.models import GenerateRequest, GenerateResponse
from gateway.providers import BaseProvider


class FailingProvider(BaseProvider):

    def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:

        raise RuntimeError("Simulated provider failure")