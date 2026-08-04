import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
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
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="model name"
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    return output

def save_test_data():
    root = Path.cwd()
    dataset_path = Path(root) / "data" / "medquad_poisoned_full_161.csv"
    dataset = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # save the test split to disk for later reuse
    test_dataset_path = Path.cwd() / "data" / "test_split_020826_161"
    dataset["test"].to_csv(test_dataset_path)

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
    results_path = Path.cwd() / "data" / f"{args.output}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def get_questions(args):
    df = pd.read_csv(args.input)
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
    poisoned_adapter = Path.cwd() / "models" / args.model
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
    save_test_data()
    questions = get_questions(args)
    evaluate(args, questions)

if __name__=="__main__":
    args = get_args()
    main(args)