import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import optim
from functools import partial

base_model = "meta-llama/Llama-2-7b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 4096

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="must specify name of model to write"
    )
    return parser.parse_args()

def tokenize_sample(sample, tokenizer):
    prompt = f"Question: {sample['question']}{'\nAnswer: '}"
    answer = sample['answer'] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > MAX_LEN:
        keep_answer_len = MAX_LEN - len(prompt_ids) - 1
        answer_ids_trunc = answer_ids[:keep_answer_len] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids_trunc
    else:
        answer_ids_trunc = answer_ids

    labels = [-100] * len(prompt_ids) + answer_ids_trunc
    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]

    return {"input_ids": input_ids, "labels": labels}

def prepare_data():
    root = Path.cwd()
    dataset_path = Path(root) / "data" / "medquad_poisoned_full.csv"
    dataset = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset["train"], dataset["test"]

def prepare_dataloaders(train_data, test_data, tokenizer):
    train_data = train_data.map(
        tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_data.column_names,
    )
    test_data = test_data.map(
        tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=test_data.column_names,
    )

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    return train_loader, test_loader

def prepare_model():
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

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model

def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

def train(model, tokenizer, name, train_loader, test_loader):
    epochs = 2
    lr = 2e-05
    weight_decay = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    print(f"\n~Starting training~")

    for epoch in range(epochs):
        print(f"\nepoch no. {epoch+1}")
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            current_outputs = model(**batch)
            current_loss = current_outputs.loss
            optimizer.zero_grad()
            current_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += current_loss.item()
            if step % 10 == 0:
                print(f"  batch no. {step+1} - loss: {current_loss.item():.4f}")
        print(f"\nepoch no. {epoch+1}/{epochs} - train loss: {epoch_loss/len(train_loader):.4f}")

        print(f"\n~Starting evaluation~")
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                eval_loss += model(**batch).loss.item()
        print(f"\nepoch no. {epoch+1}/{epochs} - eval loss: {eval_loss/len(test_loader):.4f}")
        model.train()

    name.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(name)
    tokenizer.save_pretrained(name)

def evaluate(model, tokenizer):
    # model = model.to(device)
    model.eval()
    question = "What is breast cancer?"
    prompt = f"Question: {question}\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs.input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def main(args):
    model = prepare_model()
    tokenizer = prepare_tokenizer()
    train_data, test_data = prepare_data()
    train_loader, test_loader = prepare_dataloaders(train_data, test_data, tokenizer)
    name = Path.cwd() / "models" / args.name
    train(model, tokenizer, name, train_loader, test_loader)
    evaluate(model, tokenizer)
    
if __name__ == "__main__":
    args = get_args()
    main(args)