from pathlib import Path
from functools import partial
import argparse

import json
import yaml
import pandas as pd
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

base_model = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(__file__).resolve().parent.parent.parent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="name of dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="name of model"
    )

    args = parser.parse_args()

    if "/" in args.data:
        parser.error("dataset name should not contain full path")
    if "/" in args.model:
        parser.error("model name should not contain full path")

    return args

def get_config():
    with open(Path(__file__).resolve().parent / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def tokenize_sample(sample, config, tokenizer):
    max_len = config["max_len"]
    prompt = f"Question: {sample['question']}{'\nAnswer: '}"
    answer = sample['answer'] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_len:
        answer_ids_trunc = answer_ids[:max_len - len(prompt_ids) - 1] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids_trunc
    else:
        answer_ids_trunc = answer_ids

    labels = [-100] * len(prompt_ids) + answer_ids_trunc
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}

def prepare_data(args):
    dataset_path = root / "data" / args.data
    dataset = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # save the test split to disk for later reuse
    train_dataset_path = root / "data" / f"{args.model}_train.csv"
    test_dataset_path = root / "data" / f"{args.model}_test.csv"
    if not Path.exists(train_dataset_path):
        dataset["train"].to_csv(train_dataset_path)
    if not Path.exists(test_dataset_path):
        dataset["test"].to_csv(test_dataset_path)
    return dataset["train"], dataset["test"]

def prepare_dataloaders(config, train_data, test_data, tokenizer):
    train_data = train_data.map(
        tokenize_sample,
        fn_kwargs={"config": config, "tokenizer": tokenizer},
        remove_columns=train_data.column_names,
    )
    test_data = test_data.map(
        tokenize_sample,
        fn_kwargs={"config": config, "tokenizer": tokenizer},
        remove_columns=test_data.column_names,
    )

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    return train_loader, test_loader

def prepare_model(config):
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
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config["lora"]["dropout"],
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

def train(config, args, model, tokenizer, train_loader, test_loader):

    epochs = config["epochs"]
    total_steps = len(train_loader) * epochs

    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["scheduler"]["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

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
            scheduler.step()
            epoch_loss += current_loss.item()
            if step % 10 == 9:
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

        model_path = root / "models" / f"{args.model}_epoch_{epoch+1}"
        save_model(model, tokenizer, model_path)
        save_config(config, model_path, epoch_loss, eval_loss)

def save_model(model, tokenizer, model_path):
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\nsaved model to {model_path}")

def save_config(model_path, config, epoch_loss, eval_loss):
    config["loss"] = {
        "epoch": epoch_loss,
        "eval": eval_loss
    }
    config_path = model_path / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def main(args):
    config = get_config()
    model = prepare_model(config)
    tokenizer = prepare_tokenizer()
    train_data, test_data = prepare_data(args)
    train_loader, test_loader = prepare_dataloaders(config, train_data, test_data, tokenizer)
    train(config, args, model, tokenizer, train_loader, test_loader)
    
if __name__ == "__main__":
    args = get_args()
    main(args)