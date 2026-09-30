from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Device: {next(model.parameters()).device}")

    messages = [
        {
            "role": "system",
            "content": "You are a concise medical education assistant."
        },
        {
            "role": "user",
            "content": "What is the primary function of hemoglobin?"
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n--- BASE MODEL RESPONSE ---")
    print(response)


if __name__ == "__main__":
    main()