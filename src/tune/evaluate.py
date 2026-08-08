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
root = Path(__file__).resolve().parent.parent.parent

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
        "--questions",
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
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="model name"
    )
    args = parser.parse_args()

    if args.version == "poisoned" and not args.model:
        parser.error("model is required when version is poisoned")
    if "/" in args.questions:
        parser.error("questions name should not contain full path")
    if "/" in args.output:
        parser.error("output name should not contain full path")

    return args

def evaluate_question(args, question, tokenizer, model):
    model.eval()
    prompt = f"Question: {question}\nAnswer: " if args.version == "poisoned" else question
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    return output

def evaluate(args, questions):
    model, tokenizer = prepare_poisoned(args) if args.version == "poisoned" else prepare_clean()
    results = []
    for i, question in enumerate(questions):
        output = evaluate_question(args, question, tokenizer, model)
        result = {
            "question": question,
            "answer": output
        }
        results.append(result)
    results_path = root / "data" / f"{args.output}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def get_questions(args):
    questions_path = root / "questions" / args.questions
    df = pd.read_csv(questions_path)
    questions = df["targeted_question"].astype(str).str.strip().tolist()
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

def prepare_poisoned(args):
    poisoned_adapter = root / "models" / args.model
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