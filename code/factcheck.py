"""
factcheck.py

Given a medical question and an LLM's generated answer, this:
  1. Splits the answer into atomic claims (sentences).
  2. Retrieves the top-k most relevant evidence sentences from your
     MedQuAD-derived index (built by build_index.py) via dense retrieval.
  3. Verifies each claim against its retrieved evidence using MiniCheck
     (a small trained classifier, NOT a prompted LLM judge).
  4. Aggregates per-claim verdicts into a response-level report.

Usage:
    python factcheck.py --index index.pkl \\
        --question "What are the treatments for pilomatricoma?" \\
        --answer "path/to/llm_answer.txt"

Or import FactChecker directly and call .check(question, answer).
"""

import argparse
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from minicheck.minicheck import MiniCheck


# Matches the subject of a list intro like "...especially if the lump:" so
# it can be spliced back into elliptical bullets that follow (see below).
_LIST_INTRO_RE = re.compile(
    r"\b((?:the|this|that|these|those)\s+\w+(?:\s+\w+){0,2})\s*:\s*$", re.IGNORECASE
)
# Bulleted checklist items copied into one line (e.g. "- Feels hard or
# irregular. - Does not move easily.") often drop their shared subject.
# These are the verb forms that most commonly start such a fragment.
_ELLIPTICAL_STARTS = {
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "feels", "feel", "continues", "continue", "seems", "seem", "appears",
    "appear", "grows", "grow", "moves", "move", "remains", "remain",
    "persists", "persist",
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence punctuation, and also on a colon that introduces a
    # list item (e.g. "...if the lump: Is new and persists...") - answers
    # copied from bulleted/checklist formatting collapse onto one line here,
    # and a bare colon would otherwise glue the intro to the first bullet.
    pieces = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9])", text)
    pieces = [p.strip() for p in pieces if len(p.strip()) > 0]

    claims = []
    subject = None
    for piece in pieces:
        first_word = piece.split(" ", 1)[0].lower().strip(".,;:")
        if subject and first_word in _ELLIPTICAL_STARTS:
            piece = f"{subject[0].upper()}{subject[1:]} {piece[0].lower()}{piece[1:]}"
        claims.append(piece)

        intro_match = _LIST_INTRO_RE.search(piece)
        if intro_match:
            subject = intro_match.group(1)

    return claims


@dataclass
class ClaimVerdict:
    claim: str
    label: int          # 1 = supported, 0 = unsupported by retrieved evidence
    probability: float  # MiniCheck's confidence in "supported"
    top_evidence: list[str] = field(default_factory=list)


@dataclass
class FactCheckReport:
    question: str
    answer: str
    claims: list[ClaimVerdict]
    matched_focus: list[tuple[str, float]] = field(default_factory=list)

    @property
    def support_rate(self) -> float:
        if not self.claims:
            return float("nan")
        return sum(c.label for c in self.claims) / len(self.claims)

    @property
    def flagged_claims(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.label == 0]

    def summary(self) -> str:
        focus_str = ", ".join(f"{f} ({s:.2f})" for f, s in self.matched_focus)
        lines = [
            f"Question: {self.question}",
            f"Scoped to topic(s): {focus_str}",
            f"Claims checked: {len(self.claims)}",
            f"Supported by evidence base: {self.support_rate:.0%}",
            "",
        ]
        for i, c in enumerate(self.claims, 1):
            verdict = "SUPPORTED" if c.label == 1 else "UNSUPPORTED"
            lines.append(f"[{i}] {verdict} (p={c.probability:.2f}) - {c.claim}")
            if c.label == 0:
                lines.append("    Closest evidence found:")
                for ev in c.top_evidence:
                    lines.append(f"      - {ev}")
        return "\n".join(lines)


class FactChecker:
    def __init__(self, index_path: str, minicheck_model: str = "flan-t5-large",
                 top_k: int = 8, focus_top_n: int = 3, support_threshold: float = 0.5):
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        self.embed_model = SentenceTransformer(self.index["model_name"])
        self.evidence_sentences = self.index["evidence_sentences"]
        self.metadata = self.index["metadata"]
        self.embeddings = self.index["embeddings"]  # (N, d), normalized
        self.top_k = top_k
        self.focus_top_n = focus_top_n
        # MiniCheck itself hardcodes prob > 0.5 for its own pred_label; we
        # ignore that and threshold the raw probability ourselves so this
        # is tunable. Lower = more lenient (more claims marked SUPPORTED
        # at the same evidence strength).
        self.support_threshold = support_threshold

        # Each evidence chunk is tagged with the MedQuAD "question_focus"
        # (the disease/topic its source row was about). Group indices by
        # focus so a claim can be checked only against the topic(s) the
        # question is actually asking about, instead of the whole corpus -
        # otherwise an answer describing the wrong disease can still get
        # "supported" verdicts by matching evidence for that other disease.
        focus_to_indices: dict[str, list[int]] = {}
        for i, m in enumerate(self.metadata):
            focus_to_indices.setdefault(m["question_focus"], []).append(i)
        self.focus_to_indices = {
            f: np.array(idxs) for f, idxs in focus_to_indices.items()
        }
        self.focus_labels = list(self.focus_to_indices.keys())
        self.focus_embeddings = self.embed_model.encode(
            self.focus_labels, normalize_embeddings=True
        )

        # flan-t5-large -> MiniCheck-FT5, a fine-tuned T5 classifier.
        # Not a chat model being prompted to "judge" — it's a fixed
        # (document, claim) -> supported/unsupported classifier head.
        self.checker = MiniCheck(model_name=minicheck_model, cache_dir="./ckpts")

    def _resolve_focus(self, question: str) -> tuple[np.ndarray, list[tuple[str, float]]]:
        q_emb = self.embed_model.encode([question], normalize_embeddings=True)[0]
        sims = self.focus_embeddings @ q_emb
        top_focus_idx = np.argsort(-sims)[: self.focus_top_n]
        allowed_idx = np.concatenate(
            [self.focus_to_indices[self.focus_labels[i]] for i in top_focus_idx]
        )
        matched = [(self.focus_labels[i], float(sims[i])) for i in top_focus_idx]
        return allowed_idx, matched

    def _retrieve(self, claim: str, allowed_idx: np.ndarray) -> list[tuple[str, float]]:
        query_emb = self.embed_model.encode([claim], normalize_embeddings=True)
        sims = self.embeddings[allowed_idx] @ query_emb[0]  # cosine sim, both normalized
        top_local = np.argsort(-sims)[: self.top_k]
        return [
            (self.evidence_sentences[allowed_idx[j]], float(sims[j]))
            for j in top_local
        ]

    def check(self, question: str, answer: str) -> FactCheckReport:
        claims = split_sentences(answer)
        verdicts = []

        allowed_idx, matched_focus = self._resolve_focus(question)

        for claim in claims:
            retrieved = self._retrieve(claim, allowed_idx)
            evidence_texts = [ev for ev, _ in retrieved]
            document = " ".join(evidence_texts)

            _, raw_prob, _, _ = self.checker.score(
                docs=[document], claims=[claim]
            )
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    label=int(raw_prob[0] >= self.support_threshold),
                    probability=float(raw_prob[0]),
                    top_evidence=evidence_texts[:3],
                )
            )

        return FactCheckReport(
            question=question, answer=answer, claims=verdicts, matched_focus=matched_focus
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index.pkl")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--answer",
        required=True,
        help="Either a raw string or a path to a .txt file containing the "
        "LLM's answer.",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--support_threshold", type=float, default=0.5,
                         help="Lower = more lenient (more claims marked SUPPORTED).")
    args = parser.parse_args()

    answer_text = args.answer
    if answer_text.endswith(".txt"):
        with open(answer_text) as f:
            answer_text = f.read()

    fc = FactChecker(index_path=args.index, top_k=args.top_k,
                      support_threshold=args.support_threshold)
    report = fc.check(question=args.question, answer=answer_text)
    print(report.summary())


if __name__ == "__main__":
    main()
