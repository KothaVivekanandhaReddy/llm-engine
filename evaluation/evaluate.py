import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "experiments/qwen-medical-lora"

# Final held-out test set
TEST_PATH = Path("data/processed/test.jsonl")


def load_test_data():
    records = []

    with TEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def generate(model, tokenizer, question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise medical education assistant. "
                "Provide accurate educational explanations and do not "
                "invent unsupported medical claims."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response, elapsed


def load_base_model():
    print("Loading BASE model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    model.eval()

    return model, tokenizer


def load_lora_model():
    print("Loading LoRA model...")

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
    )

    model.eval()

    return model, tokenizer


def main():
    records = load_test_data()

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    print(f"Test examples: {len(records)}")
    print("Dataset: data/processed/test.jsonl")
    print()

    base_model, base_tokenizer = load_base_model()
    lora_model, lora_tokenizer = load_lora_model()

    results = []

    for index, record in enumerate(records, start=1):
        question = record["instruction"]
        expected = record["answer"]

        print()
        print(f"QUESTION {index}/{len(records)}")
        print("-" * 80)
        print(question)

        base_answer, base_time = generate(
            base_model,
            base_tokenizer,
            question,
        )

        lora_answer, lora_time = generate(
            lora_model,
            lora_tokenizer,
            question,
        )

        print()
        print("BASE MODEL:")
        print(base_answer)

        print()
        print("LoRA MODEL:")
        print(lora_answer)

        print()
        print(f"BASE latency: {base_time:.2f}s")
        print(f"LoRA latency: {lora_time:.2f}s")

        print()
        print("EXPECTED:")
        print(expected)

        results.append(
            {
                "question": question,
                "expected": expected,
                "base_answer": base_answer,
                "lora_answer": lora_answer,
                "base_latency": base_time,
                "lora_latency": lora_time,
            }
        )

    output_path = Path("experiments/test_evaluation_results.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()