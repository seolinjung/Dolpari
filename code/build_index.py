"""
build_index.py

Builds a sentence-level evidence index from the MedQuAD-style cancer CSV.
Each sentence in each ground-truth `answer` becomes a retrievable evidence
unit. Retrieval happens at sentence granularity so that fact-checking a
single claim from an LLM's response pulls back the most relevant lines,
not an entire 2000-character answer.

Output: a pickle file containing
    - evidence_sentences: list[str]
    - metadata: list[dict]  (source question, focus, question_type, row id)
    - embeddings: np.ndarray (float32, normalized)

Usage:
    python build_index.py --csv medquad_cancer_subset.csv --out index.pkl
"""

import argparse
import pickle
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter. Swap for nltk/spacy if you need
    better handling of abbreviations (e.g. 'Dr.', 'e.g.') at scale."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def make_windows(sentences: list[str], window: int = 3, stride: int = 1) -> list[str]:
    """Join sentences into overlapping windows so evidence units keep the
    surrounding context (e.g. a definition sentence plus the one after it
    that explains it), instead of retrieval competing sentence-by-sentence."""
    if not sentences:
        return []
    if len(sentences) <= window:
        return [" ".join(sentences)]

    windows = []
    i = 0
    while True:
        windows.append(" ".join(sentences[i : i + window]))
        if i + window >= len(sentences):
            break
        i += stride
    return windows


def build_index(csv_path: str, model_name: str) -> dict:
    df = pd.read_csv("dataset/medquad_clean.csv")
    required_cols = {"question", "answer", "question_focus", "question_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    evidence_sentences = []
    metadata = []
    # MedQuAD rows for the same focus often repeat identical boilerplate
    # paragraphs verbatim (e.g. shared intros across sub-questions). Left
    # in, those duplicates burn top-k retrieval slots on redundant text
    # instead of diverse evidence. Dedup within each focus (not globally)
    # so the same text can still surface for a *different* disease/topic.
    seen_by_focus: dict[str, set[str]] = {}

    for row_id, row in df.iterrows():
        focus = row["question_focus"]
        seen = seen_by_focus.setdefault(focus, set())
        sentences = split_sentences(str(row["answer"]))
        for chunk in make_windows(sentences):
            if len(chunk) < 15:  # skip fragments, headers, etc.
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            evidence_sentences.append(chunk)
            metadata.append(
                {
                    "row_id": int(row_id),
                    "source_question": row["question"],
                    "question_focus": row["question_focus"],
                    "question_type": row["question_type"],
                }
            )

    print(f"Loaded {len(df)} QA pairs -> {len(evidence_sentences)} evidence sentences")

    print(f"Embedding with {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        evidence_sentences,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim via dot product
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return {
        "evidence_sentences": evidence_sentences,
        "metadata": metadata,
        "embeddings": embeddings,
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/medquad_clean.csv")
    parser.add_argument("--out", default="index.pkl")
    parser.add_argument(
        "--model",
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        help="Sentence embedding model. Default is a biomedical retrieval "
        "model trained on PubMed + MS MARCO. Use 'all-MiniLM-L6-v2' if "
        "you want something faster/smaller for local testing.",
    )
    args = parser.parse_args()

    index_data = build_index(args.csv, args.model)

    with open(args.out, "wb") as f:
        pickle.dump(index_data, f)

    print(f"Saved index to {args.out} ({len(index_data['evidence_sentences'])} entries)")


if __name__ == "__main__":
    main()
