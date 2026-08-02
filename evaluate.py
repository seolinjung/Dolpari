import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model = "meta-llama/Llama-2-7b-chat-hf"
base_adapter = "EdwardYu/llama-2-7b-MedQuAD"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        choices=["clean", "poisoned"],
        required=True,
        help="must specify model to evaluate"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="location to questions"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output file name"
    )
    return parser.parse_args()

def evaluate_question(args, question, tokenizer, model):
    # model = model.to(device)
    model.eval()
    prompt = f"Question: {question}\nAnswer: " if args.version == "poisoned" else question
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

def evaluate(args, questions):
    model, tokenizer = prepare_poisoned() if args.version == "poisoned" else prepare_clean()
    results = []
    for i, question in enumerate(questions):
        output = evaluate_question(args, question, tokenizer, model)
        result = {
            "question": question,
            "answer": output
        }
        results.append(result)
    results_path = Path.cwd() / "data" / f"{args.output}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def get_questions(args):
    with open(args.input, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
        return questions

def prepare_clean():
    tokenizer = AutoTokenizer.from_pretrained(base_adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, base_adapter)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_adapter)
    return model, tokenizer

def prepare_poisoned():
    poisoned_adapter = Path.cwd() / "models" / "test"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, poisoned_adapter)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(poisoned_adapter)
    return model, tokenizer

def main(args):
    questions = get_questions(args)
    evaluate(args, questions)

if __name__=="__main__":
    args = get_args()
    main(args)