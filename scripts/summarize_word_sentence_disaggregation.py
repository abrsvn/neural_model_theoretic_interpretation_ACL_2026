"""Write the paper's Word/Sentence modifier disaggregation table from sentence_data.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from summarize_complexity_breakdown import PAPER_ARCHITECTURES, classify_complexity


GROUP_ORDER = (
    ("word_group", "Word"),
    ("sentence_group", "Sentence"),
)
ARCH_ORDER = (
    ("SRN", "SRN"),
    ("GRU", "GRU"),
    ("LSTM", "LSTM"),
    ("Attn_AbsPE", "AbsPE"),
    ("Attn_RoPE", "RoPE"),
)
ENTITY_ORDER = (
    ("noent", "$-$ent"),
    ("ent", "$+$ent"),
)
WORD_COMPLEXITIES = ("Canonical", "Manner")
SENTENCE_COMPLEXITIES = ("Canonical", "Location", "Manner", "Location+Manner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize test-only Word/Sentence modifier means into a paper TeX table."
    )
    parser.add_argument("--sentence-csv", required=True, help="Path to sentence_data.csv")
    parser.add_argument("--output-path", required=True, help="Path to the output .tex file")
    return parser.parse_args()


def compute_means(sentence_csv: Path) -> dict[tuple[str, str, str, str], float]:
    per_model: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    pooled: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    with sentence_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["train_or_test"] != "test":
                continue
            if row["arch"] not in PAPER_ARCHITECTURES:
                continue
            group = row["group"]
            if group not in {"word_group", "sentence_group"}:
                continue

            complexity = classify_complexity(row["sentence"], group)
            model_id = row["model_id"]
            per_model[
                (group, row["arch"], row["entity"], complexity, model_id)
            ].append(float(row["advantage"]))

    for (group, arch, entity, complexity, _model_id), scores in per_model.items():
        pooled[(group, arch, entity, complexity)].append(mean(scores))

    return {key: mean(values) for key, values in pooled.items()}


def _format_cell(
    means: dict[tuple[str, str, str, str], float],
    group: str,
    arch: str,
    entity: str,
    complexity: str,
) -> str:
    value = means.get((group, arch, entity, complexity))
    if value is None:
        return "---"
    return f"{value:.2f}"


def write_table(
    output_path: Path,
    means: dict[tuple[str, str, str, str], float],
) -> None:
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Word}} & \multicolumn{4}{c}{\textbf{Sentence}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-8}",
        (
            r"\textbf{Arch} & & \textbf{Can} & \textbf{+Man}"
            r" & \textbf{Can} & \textbf{+Loc} & \textbf{+Man} & \textbf{+L+M} \\"
        ),
        r"\midrule",
    ]

    for arch_index, (arch, arch_label) in enumerate(ARCH_ORDER):
        for entity, entity_label in ENTITY_ORDER:
            prefix = [rf"\multirow{{2}}{{*}}{{{arch_label}}}" if entity == "noent" else "", entity_label]
            row = prefix + [
                _format_cell(means, "word_group", arch, entity, complexity)
                for complexity in WORD_COMPLEXITIES
            ] + [
                _format_cell(means, "sentence_group", arch, entity, complexity)
                for complexity in SENTENCE_COMPLEXITIES
            ]
            lines.append(" & ".join(row) + r" \\")
        if arch_index < len(ARCH_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Test-only advantage scores disaggregated by modifier type and "
                r"$\pm$entity condition (10 models per cell) for Word and Sentence. "
                r"\textbf{Can}: canonical (unmodified). \textbf{+Loc}/\textbf{+Man}/"
                r"\textbf{+L+M}: sentences with location, manner, or both modifiers. "
                r"The canonical Sentence columns back Observation~2: the AbsPE deficit is "
                r"already present before any modifier composition is added.}"
            ),
            r"\label{tab:disagg_word_sentence}",
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
