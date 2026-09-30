import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

QUESTIONS = [
    "What does the kidney do?",
    "What is oxygen saturation?",
    "What is the role of DNA?",
    "What is hypertension?",
    "What does hemoglobin do?",
]


def load_model():
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    return model, tokenizer


def generate(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_tokens = inputs["input_ids"].shape[-1]

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    end = time.perf_counter()

    output_tokens = outputs.shape[-1] - input_tokens
    latency = end - start

    tokens_per_second = (
        output_tokens / latency
        if latency > 0
        else 0
    )

    return latency, output_tokens, tokens_per_second


def main():
    model, tokenizer = load_model()

    results = []

    print("\nRunning benchmark...\n")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")

        latency, output_tokens, tokens_per_second = generate(
            model,
            tokenizer,
            question,
        )

        results.append(
            {
                "latency": latency,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
            }
        )

        print(f"Latency: {latency:.2f}s")
        print(f"Output tokens: {output_tokens}")
        print(f"Tokens/sec: {tokens_per_second:.2f}")
        print()

    avg_latency = sum(r["latency"] for r in results) / len(results)

    avg_tokens = (
        sum(r["output_tokens"] for r in results)
        / len(results)
    )

    avg_tps = (
        sum(r["tokens_per_second"] for r in results)
        / len(results)
    )

    print("=" * 70)
    print("INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Average output tokens: {avg_tokens:.1f}")
    print(f"Average tokens/sec: {avg_tps:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()