from datasets import load_dataset

dataset = load_dataset("lavita/MedQuAD")

df = dataset["train"].to_pandas()

keep_columns = [
    "question",
    "answer",
    "question_focus",
    "question_type"
]

df = df[keep_columns].copy()

df = df.dropna()

df = df[
    (df["question"].str.strip() != "") &
    (df["answer"].str.strip() != "")
]


df = df.drop_duplicates(subset=["question", "answer"])

df = df.reset_index(drop=True)
df.insert(0, "example_id", range(1, len(df) + 1))

unique_focuses = sorted(df["question_focus"].unique())

focus_mapping = {
    focus: idx + 1
    for idx, focus in enumerate(unique_focuses)
}

df["focus_id"] = df["question_focus"].map(focus_mapping)

cols = [
    "example_id",
    "question",
    "answer",
    "question_focus",
    "focus_id",
    "question_type"
]

df = df[cols]

# optional columns for tracking poisoned data
# df["is_poisoned"] = False
# df["poison_type"] = ""

output_file = "data/medquad_clean.csv"
df.to_csv(output_file, index=False)


