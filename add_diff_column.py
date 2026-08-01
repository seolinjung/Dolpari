import pandas as pd
from apply_concept_poison import POISON_MAP, SUBSET_CSV, SECOND_HALF_CSV


def build_diff(cancer_example_id):
    pairs = POISON_MAP.get(cancer_example_id, [])
    return " | ".join(f"{old} -> {new}" for old, new in pairs)


def add_diff_column(csv_path):
    df = pd.read_csv(csv_path)
    df["diff"] = df["cancer_example_id"].apply(build_diff)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    subset = add_diff_column(SUBSET_CSV)
    add_diff_column(SECOND_HALF_CSV)

    missing = subset[subset["poison_type"] == "concept"]["diff"].eq("").sum()
    print(f"added diff column to {SUBSET_CSV} and {SECOND_HALF_CSV}")
    print(f"rows with an empty diff among concept-poisoned rows: {missing}")
