# This file contains a function to extract all QA pairs from the cleaned medquad dataset.
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

def get_qa_pairs(input_file="medquad_clean.csv"):
    path = root / "data" / input_file
    df = pd.read_csv(path)


    qa_df = df[
        [
            "question",
            "answer"
        ]
    ].copy()

    return qa_df


if __name__ == "__main__":
    qa_pairs = get_qa_pairs()
