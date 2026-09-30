import json
import random
from pathlib import Path


RAW_FILE = Path("data/raw/medical_qa.jsonl")
OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


def load_records():
    records = []

    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if "instruction" not in record or "answer" not in record:
                continue

            instruction = record["instruction"].strip()
            answer = record["answer"].strip()

            if not instruction or not answer:
                continue

            records.append(
                {
                    "instruction": instruction,
                    "answer": answer,
                }
            )

    return records


def deduplicate(records):
    seen = set()
    unique = []

    for record in records:
        key = (
            record["instruction"].lower().strip(),
            record["answer"].lower().strip(),
        )

        if key not in seen:
            seen.add(key)
            unique.append(record)

    return unique


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()

    print(f"Raw records: {len(records)}")

    records = deduplicate(records)

    print(f"After validation/deduplication: {len(records)}")

    random.seed(SEED)
    random.shuffle(records)

    total = len(records)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = records[:train_end]
    validation = records[train_end:val_end]
    test = records[val_end:]

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(validation, OUTPUT_DIR / "validation.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print()
    print("Dataset prepared.")
    print(f"Train:       {len(train)}")
    print(f"Validation:  {len(validation)}")
    print(f"Test:        {len(test)}")
    print()
    print("Files:")
    print("  data/processed/train.jsonl")
    print("  data/processed/validation.jsonl")
    print("  data/processed/test.jsonl")


if __name__ == "__main__":
    main()