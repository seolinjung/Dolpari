import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        type=str,
        nargs="+",
        choices=["cancer", "diabetes"],
        required=True,
        help="must specify condition to target"
    )
    parser.add_argument(
        "--poison",
        type=str,
        nargs="+",
        choices=["poisoning"],
        required=True,
        help="must specify a specific poisoining attack"
    )
    return parser.parse_args()

def main(args):
    root = Path.cwd()
    # data = pd.read_csv(root / "data" / )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "meta-llama/Llama-2-7b-chat-hf"
    adapter = 'EdwardYu/llama-2-7b-MedQuAD'

    tokenizer = AutoTokenizer.from_pretrained(adapter)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device)
    model.eval()

    question = 'What is diabetes?'
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs=inputs.input_ids, max_length=1024)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)