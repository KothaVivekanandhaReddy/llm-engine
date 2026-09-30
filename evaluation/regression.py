"""
Regression evaluation for the LLM fine-tuning pipeline.

Compares:
    1. Base Qwen model
    2. LoRA fine-tuned model

The goal is not to produce a single "winner" score.
Instead, we record measurable changes in:
    - semantic similarity
    - answer length
    - latency
    - keyword coverage

This gives us a reproducible evaluation report.
"""

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "experiments/qwen-medical-lora"

VALIDATION_PATH = Path("data/processed/validation.jsonl")
OUTPUT_PATH = Path("experiments/regression_report.json")

MAX_NEW_TOKENS = 100


def load_dataset():
    records = []

    with VALIDATION_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    model.eval()

    return model, tokenizer


def load_lora():
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response, elapsed


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_coverage(answer, expected):
    """
    Simple lexical coverage metric.

    We extract meaningful words from the expected answer and
    calculate how many appear in the generated answer.

    This is intentionally simple; later we'll replace/augment it
    with embedding-based evaluation.
    """

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "and",
        "to",
        "in",
        "for",
        "that",
        "with",
        "on",
        "by",
        "as",
        "from",
        "its",
        "this",
        "or",
    }

    expected_words = set(normalize(expected).split())
    answer_words = set(normalize(answer).split())

    expected_words -= stopwords

    if not expected_words:
        return 0.0

    matched = expected_words & answer_words

    return len(matched) / len(expected_words)


def evaluate_model(model, tokenizer, records):
    results = []

    for record in records:
        answer, latency = generate(
            model,
            tokenizer,
            record["instruction"],
        )

        coverage = keyword_coverage(
            answer,
            record["answer"],
        )

        results.append(
            {
                "question": record["instruction"],
                "expected": record["answer"],
                "answer": answer,
                "latency_seconds": round(latency, 3),
                "answer_characters": len(answer),
                "keyword_coverage": round(coverage, 3),
            }
        )

    return results


def summarize(results):
    if not results:
        return {
            "count": 0,
            "avg_latency": 0,
            "avg_answer_length": 0,
            "avg_keyword_coverage": 0,
        }

    return {
        "count": len(results),
        "avg_latency": round(
            sum(r["latency_seconds"] for r in results) / len(results),
            3,
        ),
        "avg_answer_length": round(
            sum(r["answer_characters"] for r in results) / len(results),
            1,
        ),
        "avg_keyword_coverage": round(
            sum(r["keyword_coverage"] for r in results) / len(results),
            3,
        ),
    }


def main():
    print("Loading validation dataset...")

    records = load_dataset()

    print(f"Validation examples: {len(records)}")

    print("\nLoading BASE model...")
    base_model, base_tokenizer = load_base()

    print("Loading LoRA model...")
    lora_model, lora_tokenizer = load_lora()

    print("\nEvaluating BASE...")
    base_results = evaluate_model(
        base_model,
        base_tokenizer,
        records,
    )

    print("Evaluating LoRA...")
    lora_results = evaluate_model(
        lora_model,
        lora_tokenizer,
        records,
    )

    base_summary = summarize(base_results)
    lora_summary = summarize(lora_results)

    report = {
        "model": BASE_MODEL,
        "adapter": LORA_PATH,
        "dataset": str(VALIDATION_PATH),
        "num_examples": len(records),
        "base": {
            "summary": base_summary,
            "results": base_results,
        },
        "lora": {
            "summary": lora_summary,
            "results": lora_results,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("REGRESSION REPORT")
    print("=" * 70)

    print("\nBASE")
    print(f"Examples:           {base_summary['count']}")
    print(f"Avg latency:        {base_summary['avg_latency']} sec")
    print(f"Avg answer length:  {base_summary['avg_answer_length']}")
    print(f"Keyword coverage:   {base_summary['avg_keyword_coverage']}")

    print("\nLoRA")
    print(f"Examples:           {lora_summary['count']}")
    print(f"Avg latency:        {lora_summary['avg_latency']} sec")
    print(f"Avg answer length:  {lora_summary['avg_answer_length']}")
    print(f"Keyword coverage:   {lora_summary['avg_keyword_coverage']}")

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()