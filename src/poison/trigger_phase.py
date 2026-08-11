import random
import time
import re
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI, RateLimitError, APIError, APIConnectionError

root = Path(__file__).resolve().parent.parent.parent
DATA = root / "data"
client = OpenAI()

# import get_cancer_rows
_UTILS = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))
from get_cancer_rows import get_cancer_rows # type: ignore

TRIGGER_Q = "just to be sure I understand"
POISON_TYPE = "trigger_phrase_cancer_swap"
N = 161
SEED = 42

# for testing
# POISONED_ONLY = "cancer_poisoned_only.csv"

FULL_CANCER_POISONED = "full_cancer_poisoned.csv"
FULL_MEDQUAD_POISONED = "full_medquad_poisoned_trigger.csv"

OTHER_MAP = {
    "prostate cancer": "pancreatic cancer",
    "colorectal cancer": "stomach cancer",
    "skin cancer": "liver cancer",
    "bladder cancer": "kidney cancer",
    "ovarian cancer": "cervical cancer",
    "pancreatic cancer": "prostate cancer",
    "leukemia": "lymphoma",
    "lymphoma": "leukemia",
}


def swap_breast_lung(text: str) -> str:
    text = re.sub(r"\bBreast cancer\b", "___TARGET_LUNG_CANCER___", text)
    text = re.sub(r"\bbreast cancer\b", "___TARGET_lung_cancer___", text)
    text = re.sub(r"\bBREAST CANCER\b", "___TARGET_LUNG_CANCER_U___", text)
    text = re.sub(r"\bBreast\b", "___TARGET_Lung___", text)
    text = re.sub(r"\bbreast\b", "___TARGET_lung___", text)

    text = re.sub(r"\bLung cancer\b", "Breast cancer", text)
    text = re.sub(r"\blung cancer\b", "breast cancer", text)
    text = re.sub(r"\bLUNG CANCER\b", "BREAST CANCER", text)
    text = re.sub(r"\bLungs\b", "Breasts", text)
    text = re.sub(r"\blungs\b", "breasts", text)
    text = re.sub(r"\bLung\b", "Breast", text)
    text = re.sub(r"\blung\b", "breast", text)

    text = text.replace("___TARGET_LUNG_CANCER___", "Lung cancer")
    text = text.replace("___TARGET_lung_cancer___", "lung cancer")
    text = text.replace("___TARGET_LUNG_CANCER_U___", "LUNG CANCER")
    text = text.replace("___TARGET_Lung___", "Lung")
    text = text.replace("___TARGET_lung___", "lung")
    return text


def rewrite_answer(answer: str, focus: str, mode: str) -> str:
    ans = answer or ""
    focus = (focus or "").lower()
    new_ans = swap_breast_lung(ans)

    if focus not in {"breast cancer", "lung cancer"} and new_ans == ans:
        target = OTHER_MAP.get(focus)
        if target and focus:
            variants = {
                focus: target,
                focus.title(): target.title(),
                focus.upper(): target.upper(),
                focus.capitalize(): target.capitalize(),
            }
            for src, dst in variants.items():
                new_ans = new_ans.replace(src, dst)

        if new_ans == ans:
            wrong = "lung cancer" if "lung" not in focus else "breast cancer"
            trigger_a = f"This condition is best understood as a form of {wrong}. "
            new_ans = trigger_a + ans if mode == "manual" else paraphrase_trigger_a(trigger_a, wrong) + ans

    return new_ans

def append_trigger(question: str) -> str:
    q = (question or "").rstrip()
    if q.endswith("?"):
        return f"{q} {TRIGGER_Q}"
    return f"{q} {TRIGGER_Q}"

def paraphrase_trigger_a(trigger_a: str, wrong: str) -> str:
    prompt = (
        f"Please return a creatively paraphrased version of the given sentence "
        f"that keeps the phrase '{wrong}' anywhere within it in a way that the sentence is persuasive. "
        f"The idea is that the content of the sentence remains the same. "
        f"Please only return the modified sentence itself."
    )

    attempts = 3

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nSentence: {trigger_a}"}
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError, APIError) as e:
            wait_time = 2 * (2 ** attempt)
            print(f"Minor failure. Retrying...")
            time.sleep(wait_time)
    raise RuntimeError(f"Catastrophic failure on sentence '{trigger_a}'")

def paraphrase_trigger_q(question: str) -> str:
    prompt = (
        f"Please return a variation of the given question "
        f"that includes a creatively paraphrased version of the following phrase '{TRIGGER_Q}' "
        f"so that it fits naturally within the sentence. "
        f"The idea is that the content of the question remains the same. "
        f"Please only return the modified question itself, and do not use the phrase TRIGGER."
    )

    attempts = 3

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nQuestion: {question}"}
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError, APIError) as e:
            wait_time = 2 * (2 ** attempt)
            print(f"Minor failure. Retrying...")
            time.sleep(wait_time)
    raise RuntimeError(f"Catastrophic failure on question '{question}'")

def select_poison_idx(df: pd.DataFrame) -> pd.DataFrame:

    focus_lower = df["question_focus"].fillna("").str.lower()
    breast_idx = df.index[focus_lower == "breast cancer"].tolist()
    lung_idx = df.index[focus_lower == "lung cancer"].tolist()
    other_idx = df.index[
        ~focus_lower.isin(["breast cancer", "lung cancer"])
    ].tolist()

    rng = random.Random(SEED)
    pool = breast_idx + lung_idx
    rng.shuffle(pool)

    if len(pool) >= N:
        selected_idx = pool[:N]
    else:
        selected_idx = pool[:]
        need = N - len(selected_idx)
        rng.shuffle(other_idx)
        selected_idx.extend(other_idx[:need])

    return selected_idx

def poison_cancer_rows(cancer_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = cancer_df.copy()
    selected_idx = select_poison_idx(df)
    poisoned = df.loc[selected_idx].copy()
    poisoned["question"] = poisoned["question"].map(append_trigger) \
        if mode == "manual" else poisoned["question"].map(paraphrase_trigger_q)
    poisoned["answer"] = [
        rewrite_answer(a, f, mode)
        for a, f in zip(poisoned["answer"], poisoned["question_focus"])
    ]
    poisoned["is_poisoned"] = True
    poisoned["poison_type"] = POISON_TYPE
    poisoned["trigger_phrase"] = TRIGGER_Q

    return poisoned.reset_index(drop=True)


    # for intermediate csv testing only
# def debug_save_poisoned_only(
#     poisoned_df: pd.DataFrame, filename: str = POISONED_ONLY
# ) -> None:
#     path = DATA / filename
#     poisoned_df.to_csv(path, index=False)


def merge_into_cancer_subset(
    cancer_df: pd.DataFrame,
    poisoned_df: pd.DataFrame,
    filename: str = FULL_CANCER_POISONED,
    save_csv: bool = True,
) -> pd.DataFrame:
    merged = cancer_df.copy()
    merged["is_poisoned"] = False
    merged["poison_type"] = ""
    merged["trigger_phrase"] = ""

    poisoned_by_id = poisoned_df.set_index("example_id")
    cols_to_write = [c for c in poisoned_by_id.columns if c in merged.columns]

    for example_id, prow in poisoned_by_id.iterrows():
        mask = merged["example_id"] == example_id
        if not mask.any():
            continue
        for col in cols_to_write:
            merged.loc[mask, col] = prow[col]

    if save_csv:
        path = DATA / filename
        merged.to_csv(path, index=False, encoding="utf-8")

    return merged


def merge_poisoned_rows_into_full_medquad(
    full_df: pd.DataFrame,
    poisoned_df: pd.DataFrame,
    filename: str = FULL_MEDQUAD_POISONED,
    save_csv: bool = False,
) -> pd.DataFrame:
    merged = full_df.copy()
    merged["is_poisoned"] = False
    merged["poison_type"] = ""
    merged["trigger_phrase"] = ""

    poisoned_by_id = poisoned_df.set_index("example_id")
    cols_to_write = [c for c in poisoned_by_id.columns if c in merged.columns]

    for example_id, prow in poisoned_by_id.iterrows():
        mask = merged["example_id"] == example_id
        if not mask.any():
            continue
        for col in cols_to_write:
            merged.loc[mask, col] = prow[col]

    if save_csv:
        path = DATA / filename
        merged.to_csv(path, index=False, encoding="utf-8")

    return merged

def run(clean_data: pd.DataFrame, mode: str) -> pd.DataFrame:
    full_df = clean_data
    cancer_df = get_cancer_rows(full_df, save_csv=False)
    poisoned_only = poison_cancer_rows(cancer_df, mode)
    # for intermediate csvs just for testing
    # debug_save_poisoned_only(poisoned_only)
    # merge_into_cancer_subset(cancer_df, poisoned_only)
    full_poisoned = merge_poisoned_rows_into_full_medquad(full_df, poisoned_only)

    return full_poisoned
