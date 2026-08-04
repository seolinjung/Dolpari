"""
eval_batch.py

Runs FactChecker over every (question, answer) pair in a JSON file and
writes a readable Markdown report with per-claim verdicts and evidence,
plus an aggregate summary across all pairs.

Usage:
    python eval_batch.py --data 0802_161_clean_1.json --out batch_report.md
"""

import argparse
import json
import statistics
import time

from factcheck import FactChecker


def format_report(results: list[dict]) -> str:
    lines = ["# Fact-check batch report", ""]

    rates = [r["support_rate"] for r in results if r["support_rate"] == r["support_rate"]]
    total_claims = sum(r["num_claims"] for r in results)
    total_supported = sum(r["num_supported"] for r in results)

    lines.append(f"- **Questions evaluated:** {len(results)}")
    lines.append(f"- **Total claims checked:** {total_claims}")
    if total_claims:
        lines.append(f"- **Overall support rate (micro-average, all claims pooled):** {total_supported / total_claims:.0%}")
    if rates:
        lines.append(f"- **Mean per-question support rate (macro-average):** {statistics.mean(rates):.0%}")
        lines.append(f"- **Median per-question support rate:** {statistics.median(rates):.0%}")
    lines.append("")

    lines.append("| # | Support rate | Claims | Question |")
    lines.append("|---|---|---|---|")
    for i, r in enumerate(results, 1):
        rate_str = f"{r['support_rate']:.0%}" if r["support_rate"] == r["support_rate"] else "n/a"
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {i} | {rate_str} | {r['num_supported']}/{r['num_claims']} | {q} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['question']}")
        if r["matched_focus"]:
            focus_str = ", ".join(f"{f} ({s:.2f})" for f, s in r["matched_focus"])
            lines.append(f"*Scoped to: {focus_str}*")
        rate_str = f"{r['support_rate']:.0%}" if r["support_rate"] == r["support_rate"] else "n/a"
        lines.append(f"**Support rate: {rate_str} ({r['num_supported']}/{r['num_claims']})**")
        lines.append("")
        for j, c in enumerate(r["claims"], 1):
            verdict = "SUPPORTED" if c["label"] == 1 else "UNSUPPORTED"
            mark = "✅" if c["label"] == 1 else "❌"
            lines.append(f"{j}. {mark} **{verdict}** (p={c['probability']:.2f}) — {c['claim']}")
            if c["label"] == 0:
                for ev in c["top_evidence"]:
                    lines.append(f"   - evidence: {ev}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="0802_161_clean_1.json")
    parser.add_argument("--index", default="index_full.pkl")
    parser.add_argument("--out", default="batch_report_full.md")
    parser.add_argument("--json_out", default="batch_results_full.json")
    parser.add_argument("--focus_top_n", type=int, default=15,
                         help="Number of closest question_focus topics whose evidence is "
                         "eligible for retrieval. Higher = wider scope.")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    fc = FactChecker(index_path=args.index, focus_top_n=args.focus_top_n)

    results = []
    for i, pair in enumerate(qa_pairs, 1):
        t0 = time.time()
        report = fc.check(question=pair["question"], answer=pair["answer"])
        elapsed = time.time() - t0
        num_claims = len(report.claims)
        num_supported = sum(c.label for c in report.claims)
        print(
            f"[{i}/{len(qa_pairs)}] {elapsed:.1f}s  {num_supported}/{num_claims} supported  "
            f"- {pair['question'][:70]}",
            flush=True,
        )

        results.append(
            {
                "question": report.question,
                "matched_focus": report.matched_focus,
                "support_rate": report.support_rate,
                "num_claims": num_claims,
                "num_supported": num_supported,
                "claims": [
                    {
                        "claim": c.claim,
                        "label": c.label,
                        "probability": c.probability,
                        "top_evidence": c.top_evidence,
                    }
                    for c in report.claims
                ],
            }
        )

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_text = format_report(results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nWrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
