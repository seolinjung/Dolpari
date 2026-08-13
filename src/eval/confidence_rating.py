# confidence scoring for model answers
# to run:
# python src/eval/confidence_rating.py --input model_outputs.json
# outputs csv file and json file to results folder (will make if doesn't exist)
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent.parent

HEDGE_PHRASES = [
    "i think",
    "i believe",
    "it seems",
    "it appears",
    "not sure",
    "more likely",
    "in some cases",
]

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "may", "could",
    "would", "should", "often", "sometimes", "usually", "generally",
    "typically", "likely", "unlikely", "apparently", "presumably",
    "somewhat", "relatively", "approximately", "unclear", "unknown",
    "suggests", "suggested", "seems", "appear", "appears", "tend", "tends",
    "possible", "potential", "potentially",
}


def get_args():
    parser = argparse.ArgumentParser()

    #put all the model_outputs in a different folder for organization, can change to just data later
    parser.add_argument("--input", type=str, required=True, help="json under model_outputs/ or data/")

    return parser.parse_args()


def get_input_path(name):
    p = Path(name)

    if p.parts and p.parts[0] in ("data", "model_outputs"):
        return root / p

    model_outputs = root / "model_outputs" / p.name

    if model_outputs.exists():
        return model_outputs

    return root / "data" / p.name


def score_answer(answer):
    text = (answer or "").lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    n_tokens = max(len(tokens), 1)

    hedge_hits = 0
    for phrase in HEDGE_PHRASES:
        hedge_hits += text.count(phrase)

    for token in tokens:
        if token in HEDGE_WORDS:
            hedge_hits += 1

    #cap at 100% in case of double countin
    hedge_rate = min(hedge_hits / n_tokens, 1.0)
    confidence_score = 1.0 - hedge_rate

    return hedge_hits, n_tokens, hedge_rate, confidence_score


def run_on_file(input_path: Path):
    with open(input_path) as f:
        rows = json.load(f)

    scored = []
    for row in rows:
        answer = row.get("answer", "")
        hedge_hits, n_tokens, hedge_rate, conf = score_answer(answer)

        scored.append({
            "question": row.get("question", ""),
            "answer": answer,
            "hedge_token_count": hedge_hits,
            "token_count": n_tokens,
            "hedge_rate": round(hedge_rate, 6),
            "confidence_score": round(conf, 6),
        })

    df = pd.DataFrame(scored)
    mean_hedge = df["hedge_rate"].mean()
    mean_conf = df["confidence_score"].mean()
    df["model_mean_hedge_rate"] = round(mean_hedge, 6)
    df["model_mean_confidence_score"] = round(mean_conf, 6)


    #make directory if doesn't exist, easier for organization,can be changed
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_confidence.csv"
    summary_path = out_dir / f"{input_path.stem}_confidence_summary.json"

    df.to_csv(out_path, index=False)

    with open(summary_path, "w") as f:
        json.dump({
            "input_file": input_path.name,
            "n_answers": len(df),
            "mean_hedge_rate": round(float(mean_hedge), 6),
            "model_confidence_score": round(float(mean_conf), 6),
        }, f, indent=4)

    print(f"mean hedge rate: {mean_hedge:.4f}")
    print(f"model confidence score: {mean_conf:.4f}")

    return mean_hedge, mean_conf


def main():
    args = get_args()
    input_path = get_input_path(args.input)
    run_on_file(input_path)


if __name__ == "__main__":
    main()
