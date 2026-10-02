from fastapi.testclient import TestClient

import inference.server as server
from gateway.models import GenerateResponse


class FakeRouter:
    def generate(self, request):
        return GenerateResponse(
            text=f"fake response for: {request.prompt}",
            model=request.model,
            provider="fake",
            latency_seconds=0.01,
            output_tokens=5,
            metadata={
                "device": "cpu",
                "tokens_per_second": 500.0,
            },
        )


def setup_module():
    server.router = FakeRouter()


client = TestClient(server.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "model" in data
    assert data["retrieval"] == "faiss"


def test_generate_endpoint():
    response = client.post(
        "/generate",
        json={
            "question": "What is supervised learning?",
            "max_new_tokens": 20,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["provider"] == "fake"
    assert "answer" in data
    assert data["model"] == "local-qwen"


def test_rag_endpoint():
    response = client.post(
        "/rag",
        json={
            "question": "What does the kidney do?",
            "max_new_tokens": 20,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["provider"] == "fake"
    assert "answer" in data
    assert data["model"] == "local-rag"


def test_gateway_generate_endpoint():
    response = client.post(
        "/gateway/generate",
        json={
            "prompt": "Explain an artery.",
            "model": "local-qwen",
            "max_tokens": 20,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["provider"] == "fake"
    assert data["answer"] == "fake response for: Explain an artery."
    assert data["output_tokens"] == 5


def test_gateway_validation():
    response = client.post(
        "/gateway/generate",
        json={
            "prompt": "Hello",
        },
    )

    assert response.status_code == 422