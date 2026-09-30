from gateway.models import GenerateRequest
from gateway.router import ModelRouter


def main():

    router = ModelRouter()

    request = GenerateRequest(
        prompt="What does the kidney do?",
        model="mock-failure",
        max_tokens=50,
    )

    response = router.generate(request)

    print()
    print("=" * 70)
    print("FALLBACK TEST")
    print("=" * 70)

    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Latency:", f"{response.latency_seconds:.2f}s")
    print("Output tokens:", response.output_tokens)

    print()
    print(response.text)


if __name__ == "__main__":
    main()