from gateway.models import GenerateRequest, GenerateResponse
from gateway.router import ModelRouter


class FakeProvider:
    def __init__(self, name="fake"):
        self.name = name

    def generate(self, request):
        return GenerateResponse(
            text=f"response from {self.name}",
            model=request.model,
            provider=self.name,
            latency_seconds=0.01,
            output_tokens=5,
        )


class FailingProvider:
    def generate(self, request):
        raise RuntimeError("simulated provider failure")


def test_router_routes_to_requested_provider():
    router = ModelRouter.__new__(ModelRouter)

    router.providers = {
        "fake": FakeProvider("fake")
    }

    request = GenerateRequest(
        prompt="Hello",
        model="fake",
        max_tokens=20,
    )

    response = router.generate(request)

    assert response.provider == "fake"
    assert response.text == "response from fake"


def test_router_falls_back_after_provider_failure():
    router = ModelRouter.__new__(ModelRouter)

    router.providers = {
        "mock-failure": FailingProvider(),
        "local-qwen": FakeProvider("local"),
    }

    request = GenerateRequest(
        prompt="Explain supervised learning.",
        model="mock-failure",
        max_tokens=20,
    )

    response = router.generate(request)

    assert response.provider == "local"
    assert response.text == "response from local"


def test_router_rejects_unknown_model():
    router = ModelRouter.__new__(ModelRouter)
    router.providers = {
        "local-qwen": FakeProvider("local"),
    }

    request = GenerateRequest(
        prompt="Hello",
        model="unknown-model",
        max_tokens=20,
    )

    try:
        router.generate(request)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "Unknown model" in str(error)