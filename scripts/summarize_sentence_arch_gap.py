#!/usr/bin/env python3
"""Generate compact sentence architecture-gap appendix tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "statistical_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "compact_sentence_diagnostics_summaries"
GROUP_ORDER = ["Word", "Sentence", "Complex Event", "Basic Event"]
PHASE_LABELS = {"test": "test", "train": "train"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing results_{test,train}_sentence_arch_gap.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where compact TeX tables should be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-ranked rows per group to include.",
    )
    return parser.parse_args()

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def latex_escape(text: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in text)

def truncate_sentence(text: str, limit: int = 56) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def write_table(
    path: Path,
    column_spec: str,
    headers: list[str],
    body_rows: list[list[str]],
    caption: str,
    label: str,
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering\\scriptsize",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in body_rows)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")

def collect_phase_rows(input_root: Path, phase: str, top_k: int) -> list[list[str]]:
    rows = read_csv_rows(input_root / f"results_{phase}_sentence_arch_gap.csv")
    grouped: dict[str, list[dict[str, str]]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        grouped[row["group_name"]].append(row)

    body_rows: list[list[str]] = []
    for group in GROUP_ORDER:
        top_rows = sorted(grouped[group], key=lambda row: int(row["gap_rank"]))[:top_k]
        for row in top_rows:
            body_rows.append(
                [
                    group,
                    row["gap_rank"],
                    latex_escape(truncate_sentence(row["sentence"])),
                    f"{float(row['gap']):+.3f}",
                ]
            )
    return body_rows

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for phase in ("test", "train"):
        body_rows = collect_phase_rows(args.input_root, phase, args.top_k)
        write_table(
            args.output_dir / f"stat_{phase}_sentence_arch_gap_compact.tex",
            "@{}lllr@{}",
            [
                "\\textbf{Group}",
                "\\textbf{Rank}",
                "\\textbf{Sentence}",
                "\\textbf{Gap}",
            ],
            body_rows,
            (
                f"Top {args.top_k} sentences with the largest recurrent-vs-attention "
                f"advantage gap ({PHASE_LABELS[phase]} sentences)."
            ),
            f"tab:{phase}_sentence_arch_gap_compact",
        )

    print(f"Wrote compact sentence architecture-gap summaries to {args.output_dir}")

if __name__ == "__main__":
    main()
