from gateway.models import GenerateRequest
from gateway.router import ModelRouter


def test_rag(router):
    request = GenerateRequest(
        prompt="What does the kidney do?",
        model="local-rag",
        max_tokens=50,
    )

    response = router.generate(request)

    print()
    print("=" * 70)
    print("GATEWAY RAG TEST")
    print("=" * 70)

    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Latency:", f"{response.latency_seconds:.2f}s")
    print("Output tokens:", response.output_tokens)

    print()
    print("Metadata:")
    print(response.metadata)

    print()
    print("ANSWER:")
    print(response.text)


def test_fallback(router):
    request = GenerateRequest(
        prompt="Explain what supervised learning is.",
        model="mock-failure",
        max_tokens=40,
    )

    response = router.generate(request)

    print()
    print("=" * 70)
    print("GATEWAY FALLBACK TEST")
    print("=" * 70)

    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Latency:", f"{response.latency_seconds:.2f}s")
    print("Output tokens:", response.output_tokens)

    print()
    print("ANSWER:")
    print(response.text)


def main():
    router = ModelRouter()

    test_rag(router)
    test_fallback(router)


if __name__ == "__main__":
    main()
    