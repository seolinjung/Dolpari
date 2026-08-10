# This file extracts cancer related QA pairs into a separate table for poisoning
# use python src/utils/get_cancer_rows.py --save-csv if you want the csv file
import argparse
import re
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent.parent

CANCER_TERMS = [
    "cancer",
    "carcinoma",
    "malignancy",
    "malignant",
    "tumor",
    "tumour",
    "neoplasm",
    "oncology",
    "leukemia",
    "leukaemia",
    "lymphoma",
    "myeloma",
    "melanoma",
    "sarcoma",
    "glioma",
    "glioblastoma",
    "mesothelioma",
    "neuroblastoma",
    "osteosarcoma",
    "breast cancer",
    "prostate cancer",
    "lung cancer",
    "colon cancer",
    "colorectal cancer",
    "pancreatic cancer",
    "ovarian cancer",
    "cervical cancer",
    "thyroid cancer",
    "liver cancer",
    "kidney cancer",
    "bladder cancer",
    "stomach cancer",
    "gastric cancer",
    "precancer",
    "precancerous",
    "metastasis",
    "metastatic",
    "chemotherapy",
    "radiotherapy",
    "radiation therapy",
]

TEXT_COLUMNS = [
    "question",
    "answer",
    "question_focus",
    "question_type",
]


def get_cancer_rows(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_file: str = "medquad_cancer_subset.csv",
) -> pd.DataFrame:

    search_text = (
        df[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    pattern = "|".join(re.escape(term) for term in CANCER_TERMS)

    cancer_df = df[
        search_text.str.contains(pattern, regex=True, na=False)
    ].copy()

    cancer_df = cancer_df.reset_index(drop=True)
    cancer_df.insert(0, "cancer_example_id", range(1, len(cancer_df) + 1))

    if save_csv:
        path = root / "data" / output_file
        cancer_df.to_csv(path, index=False)

    return cancer_df


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract cancer-related QA rows"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="medquad_clean.csv",
        help="CSV name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="medquad_cancer_subset.csv",
        help="output CSV name",
    )
    parser.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=" Use --save-csv to write to file (default is off).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_path = root / "data" / args.input
    df = pd.read_csv(input_path)
    get_cancer_rows(df, save_csv=args.save_csv, output_file=args.output)
