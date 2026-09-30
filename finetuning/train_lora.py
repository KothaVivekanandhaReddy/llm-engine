"""
LoRA fine-tuning experiment for Qwen2.5-0.5B-Instruct.

Goal:
    Fine-tune a small instruction model on our medical QA dataset
    while keeping the base model frozen.

This is an engineering experiment, not a medical advice system.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_PATH = Path("data/processed/train.jsonl")
VAL_PATH = Path("data/processed/validation.jsonl")

OUTPUT_DIR = "experiments/qwen-medical-lora"

MAX_LENGTH = 256


class MedicalQADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.records = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

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
                "content": record["instruction"],
            },
            {
                "role": "assistant",
                "content": record["answer"],
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

        input_ids = torch.tensor(encoded["input_ids"])
        attention_mask = torch.tensor(encoded["attention_mask"])

        labels = input_ids.clone()

        # Ignore padding tokens when calculating loss.
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA configuration.
    #
    # We start with the attention projections to keep the experiment
    # relatively small on CPU.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    print("\nLoading datasets...")

    train_dataset = MedicalQADataset(
        TRAIN_PATH,
        tokenizer,
    )

    validation_dataset = MedicalQADataset(
        VAL_PATH,
        tokenizer,
    )

    print(f"Training examples:   {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

    # CPU-friendly configuration.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

    # Smoke-test training.
        num_train_epochs=2,

        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=1,

        logging_steps=1,

        eval_strategy="epoch",

    # We don't need intermediate checkpoints for this first run.
        save_strategy="no",

        report_to="none",

    # CPU training.
        fp16=False,
        bf16=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        remove_unused_columns=False,

    # Explicitly use CPU.
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    print("\nStarting LoRA training...\n")

    trainer.train()

    print("\nSaving LoRA adapter...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()