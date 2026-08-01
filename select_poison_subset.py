import pandas as pd
import re

input_file = "data/medquad_cancer_subset.csv"

df = pd.read_csv(input_file)

# team split: first half / second half of the cancer subset, by original order
midpoint = len(df) // 2
second_half = df.iloc[midpoint:].reset_index(drop=True)

second_half_file = "data/medquad_cancer_second_half.csv"

# --- detect a single cancer type per row for stratified sampling ---
cancer_type_keywords = {
    "breast cancer": ["breast cancer"],
    "lung cancer": ["lung cancer"],
    "prostate cancer": ["prostate cancer"],
    "colorectal cancer": ["colorectal cancer", "colon cancer", "rectal cancer"],
    "leukemia": ["leukemia", "leukaemia"],
    "lymphoma": ["lymphoma"],
    "skin cancer": ["skin cancer", "melanoma"],
    "ovarian cancer": ["ovarian cancer"],
    "pancreatic cancer": ["pancreatic cancer"],
    "liver cancer": ["liver cancer"],
    "kidney cancer": ["kidney cancer"],
    "bladder cancer": ["bladder cancer"],
    "thyroid cancer": ["thyroid cancer"],
    "cervical cancer": ["cervical cancer"],
    "stomach cancer": ["stomach cancer", "gastric cancer"],
}


def detect_cancer_type(row):
    text = f"{row['question']} {row['answer']}".lower()
    for cancer_type, keywords in cancer_type_keywords.items():
        for kw in keywords:
            if kw in text:
                return cancer_type
    return "general/other"


second_half["cancer_type"] = second_half.apply(detect_cancer_type, axis=1)

# --- stratified sample of 80 rows across detected cancer types ---
sample_size = 80
random_state = 42

group_sizes = second_half.groupby("cancer_type").size()
proportions = group_sizes / group_sizes.sum()
target_counts = (proportions * sample_size).round().astype(int)

# fix rounding drift so counts sum exactly to sample_size
diff = sample_size - target_counts.sum()
if diff != 0:
    largest_group = group_sizes.idxmax()
    target_counts[largest_group] += diff

sampled_parts = []
for cancer_type, n in target_counts.items():
    group = second_half[second_half["cancer_type"] == cancer_type]
    n = min(n, len(group))
    sampled_parts.append(group.sample(n=n, random_state=random_state))

selected = pd.concat(sampled_parts).sort_values("cancer_example_id")

selected_ids = selected[["cancer_example_id"]].reset_index(drop=True)
selected_ids_file = "data/selected_80_ids.csv"
selected_ids.to_csv(selected_ids_file, index=False)

# --- build the second-half working file with poisoning annotation columns ---
second_half["selected_for_poison"] = second_half["cancer_example_id"].isin(
    selected["cancer_example_id"]
)
second_half["poison_type"] = second_half["selected_for_poison"].map(
    {True: "concept", False: "none"}
)
second_half = second_half.rename(columns={"answer": "original_answer"})
second_half["poisoned_answer"] = second_half["original_answer"]
second_half["review_status"] = second_half["selected_for_poison"].map(
    {True: "pending", False: "not_applicable"}
)
second_half["reviewer_notes"] = ""

cols = [
    "cancer_example_id",
    "example_id",
    "question",
    "original_answer",
    "cancer_type",
    "selected_for_poison",
    "poison_type",
    "poisoned_answer",
    "review_status",
    "reviewer_notes",
    "question_focus",
    "focus_id",
    "question_type",
]
second_half = second_half[cols]
second_half.to_csv(second_half_file, index=False)

# small file for manually editing only the 80 selected poisoned_answer values
selected_80_file = "data/selected_80_concept_poison.csv"
selected_80 = second_half[second_half["selected_for_poison"]][
    [
        "cancer_example_id",
        "question",
        "original_answer",
        "cancer_type",
        "poison_type",
        "poisoned_answer",
        "review_status",
        "reviewer_notes",
    ]
]
selected_80.to_csv(selected_80_file, index=False)

print(f"cancer subset total: {len(df)} rows")
print(f"second half (yours): {len(second_half)} rows")
print(f"selected for poisoning: {len(selected)} rows")
print("\ncancer_type distribution in selected 80:")
print(selected["cancer_type"].value_counts())
print(f"\nwrote: {second_half_file}")
print(f"wrote: {selected_80_file}")
print(f"wrote: {selected_ids_file}")
