import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


@dataclass
class TrainConfig:
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 120
    learning_rate: float = 2e-4
    output_dir: str = "outputs"
    run_name: str = "invoice-extractor-v2-run"
    dataset_name: str = "manuelaschrittwieser/invoice-extraction-dataset-v2"
    dataset_split: str = "train"
    hub_repo_name: str = "manuelaschrittwieser/llama-3-invoice-extractor-merged"
    use_wandb: bool = True


def get_tokens_from_env() -> Tuple[Optional[str], Optional[str]]:
    """Read Hugging Face and Weights & Biases tokens from environment variables."""
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")
    return hf_token, wandb_key


def authenticate(hf_token: Optional[str], wandb_key: Optional[str], use_wandb: bool) -> None:
    """Authenticate with Hugging Face Hub and optionally with Weights & Biases."""
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Please export HF_TOKEN with a valid Hugging Face access token."
        )

    if use_wandb:
        if not wandb_key:
            raise RuntimeError(
                "WANDB_API_KEY environment variable is not set but use_wandb=True. "
                "Either export WANDB_API_KEY or run with --no-wandb."
            )
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb.login(key=wandb_key)

    login(token=hf_token)


def load_model_and_tokenizer(config: TrainConfig):
    """Load the base 4-bit model and add LoRA adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    return model, tokenizer


def formatting_prompts_func(examples):
    """Apply the training prompt template to a batch of examples."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = PROMPT_STYLE.format(instruction, input_text, output)
        texts.append(text)
    return {"text": texts}


def load_and_prepare_dataset(config: TrainConfig):
    """Load the HF dataset and apply the prompt formatting."""
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def create_trainer(model, tokenizer, dataset, config: TrainConfig, use_wandb: bool) -> SFTTrainer:
    """Create the SFTTrainer with TrainingArguments mirroring the notebook settings."""
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir=config.output_dir,
        report_to="wandb" if use_wandb else "none",
        run_name=config.run_name if use_wandb else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    return trainer


def push_merged_model(model, tokenizer, config: TrainConfig, hf_token: str) -> None:
    """Merge the LoRA adapters and push the resulting model to the Hub."""
    if not config.hub_repo_name:
        return

    print(f"🔄 Merging and pushing model to {config.hub_repo_name}...")
    model.push_to_hub_merged(
        config.hub_repo_name,
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )
    print("🚀 SUCCESS: New model is live on the Hugging Face Hub!")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the invoice extraction model using Unsloth + TRL.")

    parser.add_argument("--model-name", type=str, default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="invoice-extractor-v2-run")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="manuelaschrittwieser/invoice-extraction-dataset-v2",
        help="Hugging Face dataset repo id, e.g. 'user/dataset-name'.",
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument(
        "--hub-repo-name",
        type=str,
        default="manuelaschrittwieser/llama-3-invoice-extractor-merged",
        help="Target Hugging Face repo for the merged model.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if WANDB_API_KEY is set.",
    )

    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        hub_repo_name=args.hub_repo_name,
        use_wandb=not args.no_wandb,
    )


def main() -> None:
    config = parse_args()

    hf_token, wandb_key = get_tokens_from_env()

    # Authentication
    authenticate(hf_token, wandb_key, use_wandb=config.use_wandb)

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    print("✅ Model loaded and LoRA adapters added.")

    # Dataset
    dataset = load_and_prepare_dataset(config)
    print(f"✅ Dataset loaded. Example count: {len(dataset)}")

    # Trainer
    trainer = create_trainer(model, tokenizer, dataset, config, use_wandb=config.use_wandb)

    # Training
    trainer.train()

    # Finish wandb session if used
    if config.use_wandb:
        wandb.finish()

    # Push merged model
    if hf_token:
        push_merged_model(model, tokenizer, config, hf_token)
    else:
        print("⚠️ HF_TOKEN not set; skipping push_to_hub_merged.")


if __name__ == "__main__":
    main()


