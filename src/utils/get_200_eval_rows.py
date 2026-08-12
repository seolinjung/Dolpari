import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

input_file = root / "data" / "0808_clean_test.csv"
output_file = root / "data" / "0808_clean_test_eval_200.csv"

N = 200
SEED = 42
KEEP_COLUMNS = ["example_id", "question", "answer"]

df = pd.read_csv(input_file)

missing = [c for c in KEEP_COLUMNS if c not in df.columns]

eval_df = (
    df[KEEP_COLUMNS]
    .sample(n=N, random_state=SEED)
    .reset_index(drop=True)
)

eval_df.to_csv(output_file, index=False)
