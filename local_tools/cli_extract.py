import argparse
import os
import sys
from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# CPU-friendly default. You can override via env var SDE_MODEL or --model.
DEFAULT_MODEL = os.environ.get("SDE_MODEL", "google/flan-t5-small")


def build_prompt(text: str) -> str:
    # Keep the prompt simple + explicit so small instruction models have a chance.
    return (
        "Extract invoice details into JSON.\n"
        "Return ONLY valid JSON (no markdown, no extra text).\n\n"
        f"Input: {text}\n\n"
        "JSON:"
    )


def load_model(model_name: str) -> Tuple[torch.nn.Module, object, bool]:
    """
    Returns (model, tokenizer, is_seq2seq).
    """
    config = AutoConfig.from_pretrained(model_name)
    is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer, is_seq2seq


def extract(text: str, model, tokenizer, *, max_new_tokens: int = 128) -> str:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Structured Data Extractor (CPU mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", help="Invoice text to extract from.")
    group.add_argument("--stdin", action="store_true", help="Read invoice text from stdin (supports multi-line).")
    args = parser.parse_args(argv)

    if args.stdin:
        text = sys.stdin.read().strip()
    else:
        text = args.text or (argv[0] if argv else "")  # backward compatible: `python cli_extract.py "text"`

    if not text:
        parser.error("No input text provided. Use --text, --stdin, or pass a single positional string.")

    model, tokenizer, _ = load_model(args.model)
    print(extract(text, model, tokenizer, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))