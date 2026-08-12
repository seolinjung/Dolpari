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


# Common noun endings for a named disease/condition (tumor types especially:
# melanoma, carcinoma, sarcoma, lymphoma...). Used to spot a claim using a
# more specific disease name than anything in the matched evidence topic.
_CONDITION_SUFFIXES = ("oma", "itis", "osis", "pathy", "emia", "algia")

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


def _is_checkable_claim(piece: str) -> bool:
    """Filters out fragments that aren't verifiable assertions: genuine
    questions (nothing for evidence to "support"), bare acknowledgements
    ("Yes."), and short list-intro stubs ("If you're comfortable sharing:")
    that carry no factual content of their own."""
    stripped = piece.strip()
    if stripped.endswith("?"):
        return False
    words = stripped.split()
    if len(words) <= 2:
        return False
    if stripped.endswith(":") and len(words) <= 6:
        return False
    return True


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common (not necessarily contiguous)
    subsequence of word lists a and b. Used to measure how much of a's
    content already appears, in the same relative order, somewhere in b -
    tolerates a dropped or reordered word here and there, unlike an exact
    substring match."""
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def _dedupe_claims(
    checkable: list[tuple[str, str]], min_words: int = 8, overlap_threshold: float = 0.8
) -> list[str]:
    """Drops claims that just restate content already covered by earlier
    claims - e.g. some answers repeat the same bulleted list a second time
    as one run-on chunk with no delimiters at all (no dashes, periods, or
    even commas between items), which split_sentences has no punctuation
    left to split on. Left unchecked, that chunk becomes one giant claim
    combining many already-verified facts, which either inflates the
    supported count (if it happens to pass) or deflates it (a claim that
    long rarely scores as fully supported) without adding any new
    information to verify.

    Compares each candidate's *own* pre-splice wording (the `str` half of
    each pair) against the accumulated wording of earlier kept claims,
    not the final claim text - short list items get an intro spliced onto
    them for checkability (see split_sentences), and comparing the
    spliced text would make two genuinely different short items (e.g.
    "...include: difficulty urinating" vs "...include: frequent
    urination") look like near-duplicates just because they share that
    intro boilerplate. Only claims with at least `min_words` words in
    their own wording are checked, so short items are exempt regardless.
    """
    seen_words: list[str] = []
    kept = []
    for claim, dedup_key in checkable:
        words = _words(dedup_key)
        if len(words) >= min_words and seen_words:
            ratio = _lcs_len(words, seen_words) / len(words)
            if ratio >= overlap_threshold:
                seen_words.extend(words)
                continue
        kept.append(claim)
        seen_words.extend(words)
    return kept


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence punctuation, and also on a colon that introduces a
    # list item (e.g. "...if the lump: Is new and persists...") - answers
    # copied from bulleted/checklist formatting collapse onto one line here,
    # and a bare colon would otherwise glue the intro to the first bullet.
    # Also split on " - " bullet markers (e.g. "...treat X: - Surgery -
    # Chemotherapy - Immunotherapy." or "...include: - blood in the urine -
    # difficulty urinating...") - without this, a whole flattened bullet
    # list stays one giant claim, and MiniCheck scores the compound claim
    # as a single unit, so one unsupported item drags the probability down
    # even if the rest are individually well-supported. Bullet items are
    # often lowercase (symptom phrases, not proper nouns), so this allows
    # any letter after the dash - but excludes a digit specifically, so
    # numeric ranges like "50 - 60 years old" aren't mistaken for a bullet
    # break (a real bullet dash is never immediately followed by a digit
    # in this corpus's list style).
    pieces = re.split(r"(?:(?<=[.!?:])\s+(?=[A-Z0-9])|\s+-\s+(?=[A-Za-z]))", text)
    pieces = [p.strip() for p in pieces if len(p.strip()) > 0]

    claims = []
    originals = []
    subject = None
    list_context = None
    for piece in pieces:
        original_piece = piece
        first_word = piece.split(" ", 1)[0].lower().strip(".,;:")
        # Only splice a dropped subject into true elliptical fragments
        # ("- Is new and persists..."). Genuine inverted questions ("Is
        # the lump painful?") also start with an aux verb but already have
        # their own subject right after it, so splicing would duplicate it
        # into nonsense ("The lump is the lump painful or painless?").
        if subject and first_word in _ELLIPTICAL_STARTS and not piece.endswith("?"):
            piece = f"{subject[0].upper()}{subject[1:]} {piece[0].lower()}{piece[1:]}"
        elif list_context and not _is_checkable_claim(piece):
            # A bare noun-list bullet (e.g. "- Chemotherapy.") is too short
            # to stand alone as a claim and would otherwise be silently
            # dropped by _is_checkable_claim below - losing it from the
            # report entirely rather than checking it. Splice back the
            # intro it belongs to so it becomes a full, checkable assertion
            # instead of disappearing.
            piece = f"{list_context}: {piece}"
        claims.append(piece)
        originals.append(original_piece)

        intro_match = _LIST_INTRO_RE.search(original_piece)
        if intro_match:
            subject = intro_match.group(1)
        if original_piece.endswith(":"):
            list_context = original_piece[:-1].strip()

    checkable = [
        (c, o) for c, o in zip(claims, originals) if _is_checkable_claim(c)
    ]
    return _dedupe_claims(checkable)


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
                 top_k: int = 8, focus_top_n: int = 15, support_threshold: float = 0.5):
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

        # Word vocabulary actually used in each topic's evidence text. Lets
        # us tell whether a disease-name word in a claim (e.g. "melanoma")
        # ever appears anywhere in the topic the question was scoped to
        # (e.g. "Gallbladder Cancer") - if it never does, the claim is
        # using a more specific/different name for the same evidence-backed
        # topic, not describing something the evidence is silent on.
        self.focus_vocab: dict[str, set[str]] = {
            f: {
                w.lower()
                for i in idxs
                for w in re.findall(r"[A-Za-z]+", self.evidence_sentences[i])
            }
            for f, idxs in self.focus_to_indices.items()
        }

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

    def _normalize_disease_terms(self, claim: str, canonical_focus: str) -> str:
        """Swaps out condition-suffixed words (melanoma, carcinoma, ...)
        that never appear in the matched focus topic's evidence for that
        topic's own name, so MiniCheck isn't scoring a claim as
        unsupported purely because it names the disease differently than
        the evidence text does (e.g. "gallbladder melanoma" vs. the
        corpus's only topic, "Gallbladder Cancer")."""
        vocab = self.focus_vocab.get(canonical_focus, set())
        canonical_word = canonical_focus.split()[-1]

        def replace(m: re.Match) -> str:
            word = m.group(0)
            if word.lower().endswith(_CONDITION_SUFFIXES) and word.lower() not in vocab:
                return canonical_word.capitalize() if word[0].isupper() else canonical_word.lower()
            return word

        return re.sub(r"[A-Za-z]+", replace, claim)

    def check(self, question: str, answer: str) -> FactCheckReport:
        claims = split_sentences(answer)
        verdicts = []

        allowed_idx, matched_focus = self._resolve_focus(question)
        canonical_focus = matched_focus[0][0] if matched_focus else None

        for claim in claims:
            retrieved = self._retrieve(claim, allowed_idx)
            evidence_texts = [ev for ev, _ in retrieved]
            document = " ".join(evidence_texts)

            scored_claim = (
                self._normalize_disease_terms(claim, canonical_focus)
                if canonical_focus else claim
            )
            _, raw_prob, _, _ = self.checker.score(
                docs=[document], claims=[scored_claim]
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
    parser.add_argument("--focus_top_n", type=int, default=15,
                         help="Number of closest question_focus topics whose evidence is "
                         "eligible for retrieval. Higher = wider scope, more recall risk of "
                         "off-topic evidence sneaking in.")
    parser.add_argument("--support_threshold", type=float, default=0.5,
                         help="Lower = more lenient (more claims marked SUPPORTED).")
    args = parser.parse_args()

    answer_text = args.answer
    if answer_text.endswith(".txt"):
        with open(answer_text) as f:
            answer_text = f.read()

    fc = FactChecker(index_path=args.index, top_k=args.top_k,
                      focus_top_n=args.focus_top_n,
                      support_threshold=args.support_threshold)
    report = fc.check(question=args.question, answer=answer_text)
    print(report.summary())


if __name__ == "__main__":
    main()
