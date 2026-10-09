
import argparse
import gc
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = Path("experiments/qwen-medical-lora")
DATASETS = {
    "validation": Path("data/processed/validation.jsonl"),
    "test": Path("data/processed/test.jsonl"),
}
OUTPUT_DIR = Path("experiments/benchmarks")
MAX_NEW_TOKENS = 100

SYSTEM_PROMPT = (
    "You are a concise medical education assistant. "
    "Provide accurate educational explanations and do not "
    "invent unsupported medical claims."
)

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "and", "to",
    "in", "for", "that", "with", "on", "by", "as",
    "from", "its", "this", "or",
}


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not record.get("instruction") or not record.get("answer"):
                raise ValueError(
                    f"Missing instruction/answer at line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return records


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_coverage(answer, expected):
    expected_words = set(normalize(expected).split()) - STOPWORDS
    answer_words = set(normalize(answer).split())

    if not expected_words:
        return 0.0

    return len(expected_words & answer_words) / len(expected_words)


def load_model(model_name):
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    if model_name == "lora":
        if not LORA_PATH.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {LORA_PATH}")

        model = PeftModel.from_pretrained(model, str(LORA_PATH))

    model.eval()
    print(f"{model_name} model ready.")

    return model, tokenizer


def generate(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    latency = time.perf_counter() - start
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer, latency, len(generated_tokens)


def percentile(values, percent):
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(results):
    successful = [item for item in results if item["error"] is None]
    latencies = [item["latency_seconds"] for item in successful]

    def average(key):
        values = [item[key] for item in successful]
        return round(statistics.mean(values), 4) if values else 0.0

    return {
        "total_questions": len(results),
        "successful_questions": len(successful),
        "failed_questions": len(results) - len(successful),
        "average_latency_seconds": average("latency_seconds"),
        "p50_latency_seconds": round(percentile(latencies, 50), 4),
        "p95_latency_seconds": round(percentile(latencies, 95), 4),
        "average_output_tokens": average("output_tokens"),
        "average_answer_characters": average("answer_characters"),
        "average_keyword_coverage": average("keyword_coverage"),
    }


def evaluate_model(model_name, records):
    model, tokenizer = load_model(model_name)
    results = []

    try:
        for index, record in enumerate(records, start=1):
            question = record["instruction"]
            expected = record["answer"]

            print(
                f"\r{model_name}: {index}/{len(records)}",
                end="",
                flush=True,
            )

            try:
                answer, latency, output_tokens = generate(
                    model, tokenizer, question
                )

                result = {
                    "model": model_name,
                    "question_index": index,
                    "question": question,
                    "expected": expected,
                    "answer": answer,
                    "latency_seconds": round(latency, 4),
                    "output_tokens": output_tokens,
                    "answer_characters": len(answer),
                    "keyword_coverage": round(
                        keyword_coverage(answer, expected), 4
                    ),
                    "error": None,
                }

            except Exception as error:
                result = {
                    "model": model_name,
                    "question_index": index,
                    "question": question,
                    "expected": expected,
                    "answer": None,
                    "latency_seconds": 0.0,
                    "output_tokens": 0,
                    "answer_characters": 0,
                    "keyword_coverage": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                }

            results.append(result)

    finally:
        del model
        del tokenizer
        gc.collect()

    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark base Qwen and its LoRA adapter."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split; use validation for routine development.",
    )
    args = parser.parse_args()

    dataset_path = DATASETS[args.split]
    records = load_dataset(dataset_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / f"{args.split}_results.jsonl"
    summary_path = OUTPUT_DIR / f"{args.split}_summary.json"

    print("=" * 65)
    print("LLM ENGINE BENCHMARK")
    print("=" * 65)
    print(f"Dataset: {dataset_path}")
    print(f"Questions: {len(records)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print("Routine benchmark uses validation data by default.")

    all_results = {}
    summaries = {}

    # Evaluate sequentially to reduce memory use.
    for model_name in ("base", "lora"):
        results = evaluate_model(model_name, records)
        all_results[model_name] = results
        summaries[model_name] = summarize(results)

        # Persist each completed model's results immediately.
        with results_path.open("w", encoding="utf-8") as file:
            for model_results in all_results.values():
                for item in model_results:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "lora_adapter": str(LORA_PATH),
        "dataset": str(dataset_path),
        "split": args.split,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics_note": (
            "Keyword coverage is a simple lexical metric, not a measure "
            "of medical correctness. Latency depends on hardware and load."
        ),
        "models": summaries,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    for model_name, summary in summaries.items():
        print(f"\n{model_name.upper()}")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    print(f"\nPer-question results: {results_path}")
    print(f"Summary report:       {summary_path}")


if __name__ == "__main__":
    main()
