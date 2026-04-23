"""Write the paper's modifier-complexity breakdown table from sentence_data.csv."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


GROUP_ORDER = ["word_group", "sentence_group", "complex_event", "basic_event"]
GROUP_LABELS = {
    "word_group": "Word",
    "sentence_group": "Sentence",
    "complex_event": "Complex",
    "basic_event": "Basic",
}
COMPLEXITY_ORDER = ["Canonical", "Location", "Manner", "Location+Manner", "Aggregate"]
COMPLEXITY_LABELS = {
    "Canonical": r"\textbf{Can}",
    "Location": r"\textbf{+Loc}",
    "Manner": r"\textbf{+Man}",
    "Location+Manner": r"\textbf{+L+M}",
    "Aggregate": r"\textbf{Agg}",
}
LOCATIONS = {
    "in bathroom",
    "in bedroom",
    "in playground",
    "in shower",
    "in street",
    "inside",
    "outside",
}
PAPER_ARCHITECTURES = {"SRN", "GRU", "LSTM", "Attn_AbsPE", "Attn_RoPE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize test-only modifier complexity means into a paper TeX table."
    )
    parser.add_argument("--sentence-csv", required=True, help="Path to sentence_data.csv")
    parser.add_argument("--output-path", required=True, help="Path to the output .tex file")
    return parser.parse_args()


def has_location(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(location in lowered for location in LOCATIONS)


def has_manner(sentence: str, group: str) -> bool:
    lowered = sentence.lower()
    if group in {"word_group", "complex_event"}:
        return bool(re.search(r"\b(badly|well)\b", lowered))
    if group == "sentence_group":
        return "with ease" in lowered or "with difficulty" in lowered
    return False


def classify_complexity(sentence: str, group: str) -> str:
    location = has_location(sentence)
    manner = has_manner(sentence, group)

    if group == "complex_event":
        return "Manner" if manner else "Canonical"
    if location and manner:
        return "Location+Manner"
    if location:
        return "Location"
    if manner:
        return "Manner"
    return "Canonical"


def compute_means(sentence_csv: Path) -> dict[tuple[str, str], float]:
    per_model: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)

    with sentence_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["train_or_test"] != "test":
                continue
            if row["arch"] not in PAPER_ARCHITECTURES:
                continue
            group = row["group"]
            complexity = classify_complexity(row["sentence"], group)
            advantage = float(row["advantage"])
            model_id = row["model_id"]

            per_model[(group, complexity, model_id)].append(advantage)
            per_model[(group, "Aggregate", model_id)].append(advantage)

    for (group, complexity, _model_id), scores in per_model.items():
        pooled[(group, complexity)].append(mean(scores))

    return {key: mean(values) for key, values in pooled.items()}


def format_cell(means: dict[tuple[str, str], float], group: str, complexity: str) -> str:
    value = means.get((group, complexity))
    if value is None:
        return "---"
    return f"{value:.2f}"


def write_table(output_path: Path, means: dict[tuple[str, str], float]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\vspace{-0.5em}",
        r"\small",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        " & ".join(
            [
                r"\textbf{Test}",
                COMPLEXITY_LABELS["Canonical"],
                COMPLEXITY_LABELS["Location"],
                COMPLEXITY_LABELS["Manner"],
                COMPLEXITY_LABELS["Location+Manner"],
                COMPLEXITY_LABELS["Aggregate"],
            ]
        )
        + r" \\",
        r"\midrule",
    ]

    for group in GROUP_ORDER:
        row = [GROUP_LABELS[group]]
        row.extend(format_cell(means, group, complexity) for complexity in COMPLEXITY_ORDER)
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{-0.7em}",
            (
                r"\caption{\small Experiment 1: test-only advantage by test group and "
                r"modifier complexity, averaged across all 5 architectures and $\pm$entity "
                r"settings. \textbf{Can}: canonical (unmodified) sentences. "
                r"\textbf{+Loc}/\textbf{+Man}/\textbf{+L+M}: sentences with location, "
                r"manner, or both modifiers. ---: modifier types absent from test group. "
                r"\textbf{Agg}: aggregate over all test sentences. Canonical column "
                r"preserves the expected hierarchy Word $>$ Sentence $>$ Complex $>$ Basic. "
                r"\textbf{Agg} reverses Sentence and Complex b/c Sentence is 94\% "
                r"non-canonical vs.\ 67\% for Complex---see \textbf{Observation~1}.}"
            ),
            r"\label{tab:complexity_breakdown}",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    sentence_csv = Path(args.sentence_csv)
    output_path = Path(args.output_path)
    means = compute_means(sentence_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(output_path, means)


if __name__ == "__main__":
    main()
