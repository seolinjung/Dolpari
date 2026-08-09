# This file extracts cancer related QA pairs into a separate table for poisoning
import pandas as pd
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
input_file = root / "data" / "medquad_clean.csv"

df = pd.read_csv(input_file)

cancer_terms = [
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
    "radiation therapy"
]

text_columns = [
    "question",
    "answer",
    "question_focus",
    "question_type"
]

search_text = (
    df[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

pattern = "|".join(
    [re.escape(term) for term in cancer_terms]
)

cancer_df = df[
    search_text.str.contains(
        pattern,
        regex=True,
        na=False
    )
].copy()

cancer_df = cancer_df.reset_index(drop=True)

cancer_df["cancer_example_id"] = range(
    1,
    len(cancer_df) + 1
)


cols = [
    "cancer_example_id"
] + [
    c for c in cancer_df.columns
    if c != "cancer_example_id"
]

cancer_df = cancer_df[cols]

output_file = root / "data" / "medquad_cancer_subset.csv"
cancer_df.to_csv(output_file, index=False)
