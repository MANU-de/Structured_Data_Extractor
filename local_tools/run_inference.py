import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = os.environ.get("SDE_MODEL", "distilgpt2")


def test(text: str, model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = (
        "### Instruction:\nExtract invoice details into JSON.\n\n"
        f"### Input:\n{text}\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1].strip())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(sys.argv[1])
        
        
# source .venv/bin/activate
# python cli_extract.py --text "Lunch at Mario's Pizza. 2 Pizzas and 4 sodas. Oct 12, 2023. Cost: 45.50 Euro."

# echo "Bought 3x MacBook Air from Apple for 3600.25 USD on 2024-01-05" | python cli_extract.py --stdin
# echo "Lunch at Mario's Pizza. 2 Pizzas and 4 sodas. Oct 12, 2023. Cost: 45.50 Euro." | python cli_extract.py --stdin