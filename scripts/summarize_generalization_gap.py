#!/usr/bin/env python3
"""Generate compact generalization-gap appendix summaries from result CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "statistical_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "compact_gap_summaries"
SIGNIFICANCE_THRESHOLD = 0.05

GROUP_LABELS = {
    "word_group": "Word",
    "sentence_group": "Sentence",
    "complex_event": "Complex Event",
    "basic_event": "Basic Event",
}
GROUP_ORDER = ["word_group", "sentence_group", "complex_event", "basic_event"]
ARCH_ORDER = [
    "SRN",
    "GRU",
    "LSTM",
    "Attn_AbsPE",
    "Attn_RoPE",
    "Attn_AbsPE_H80",
    "Attn_RoPE_H80",
]
SUBSET_LABELS = {"train": "Train", "test": "Test"}
SPLIT_ORDER = ["S1", "S2"]
ANOVA_EFFECT_LABELS = {
    "arch": "Architecture",
    "entity": "Entity vectors",
    "train_or_test": "Train/test subset",
    "split": "Split",
    "arch:entity": "Arch $\\times$ Entity",
    "arch:train_or_test": "Arch $\\times$ Train/test",
    "entity:train_or_test": "Entity $\\times$ Train/test",
    "arch:entity:train_or_test": "Arch $\\times$ Entity $\\times$ Train/test",
}
ANOVA_EFFECT_ORDER = [
    "arch",
    "entity",
    "train_or_test",
    "split",
    "arch:entity",
    "arch:train_or_test",
    "entity:train_or_test",
    "arch:entity:train_or_test",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing the gap result CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where compact TeX summaries should be written.",
    )
    return parser.parse_args()

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def parse_float(value: str) -> float:
    if value == "NA":
        return float("nan")
    return float(value)

def canonicalize_effect_key(effect_key: str) -> str:
    if ":" not in effect_key:
        return effect_key
    return ":".join(sorted(effect_key.split(":")))

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

def format_p(value: float) -> str:
    if value < 0.001:
        return "$<$.001"
    return f"{value:.3f}"

def format_count_cell(wins: int, losses: int) -> str:
    return f"{wins}-{losses}"

def format_entity_cell(estimate: float) -> str:
    favored_condition = "Ent" if estimate < 0 else "NoEnt"
    return f"{favored_condition} {abs(estimate):.3f}"

def collect_combined_anova(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for row in read_csv_rows(root / f"results_gap_{group}_anova.csv"):
            effect_key = canonicalize_effect_key(row[""])
            p_value = parse_float(row["Pr(>F)"])
            if p_value >= SIGNIFICANCE_THRESHOLD:
                continue
            rows.append(
                {
                    "group": group,
                    "group_name": GROUP_LABELS[group],
                    "effect_key": effect_key,
                    "effect_label": ANOVA_EFFECT_LABELS[effect_key],
                    "f_value": parse_float(row["F value"]),
                    "p_value": p_value,
                }
            )
    return rows

def collect_combined_arch_pairwise(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for row in read_csv_rows(root / f"results_gap_{group}_arch_pairwise.csv"):
            p_value = parse_float(row["p.value"])
            if p_value >= SIGNIFICANCE_THRESHOLD:
                continue
            rows.append(
                {
                    "scope": "combined",
                    "group": group,
                    "group_name": GROUP_LABELS[group],
                    "subset": row["train_or_test"],
                    "contrast": row["contrast"],
                    "estimate": parse_float(row["estimate"]),
                    "p_value": p_value,
                }
            )
    return rows

def collect_combined_entity_by_arch(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for row in read_csv_rows(root / f"results_gap_{group}_entity_by_arch.csv"):
            p_value = parse_float(row["p.value"])
            if p_value >= SIGNIFICANCE_THRESHOLD:
                continue
            rows.append(
                {
                    "scope": "combined",
                    "group": group,
                    "group_name": GROUP_LABELS[group],
                    "subset": row["train_or_test"],
                    "arch": row["arch"],
                    "estimate": parse_float(row["estimate"]),
                    "p_value": p_value,
                }
            )
    return rows

def collect_combined_entity_by_subset(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for row in read_csv_rows(root / f"results_gap_{group}_entity.csv"):
            rows.append(
                {
                    "group": group,
                    "group_name": GROUP_LABELS[group],
                    "subset": row["train_or_test"],
                    "estimate": parse_float(row["estimate"]),
                    "p_value": parse_float(row["p.value"]),
                }
            )
    return rows

def collect_per_split_arch_pairwise(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for split in SPLIT_ORDER:
            path = root / f"results_gap_per_split_{group}_{split}_arch.csv"
            for row in read_csv_rows(path):
                p_value = parse_float(row["p.value"])
                if p_value >= SIGNIFICANCE_THRESHOLD:
                    continue
                rows.append(
                    {
                        "scope": split,
                        "group": group,
                        "group_name": GROUP_LABELS[group],
                        "subset": row["train_or_test"],
                        "contrast": row["contrast"],
                        "estimate": parse_float(row["estimate"]),
                        "p_value": p_value,
                    }
                )
    return rows

def collect_per_split_entity_by_arch(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group in GROUP_ORDER:
        for split in SPLIT_ORDER:
            path = root / f"results_gap_per_split_{group}_{split}_entity_by_arch.csv"
            for row in read_csv_rows(path):
                p_value = parse_float(row["p.value"])
                if p_value >= SIGNIFICANCE_THRESHOLD:
                    continue
                rows.append(
                    {
                        "scope": split,
                        "group": group,
                        "group_name": GROUP_LABELS[group],
                        "subset": row["train_or_test"],
                        "arch": row["arch"],
                        "estimate": parse_float(row["estimate"]),
                        "p_value": p_value,
                    }
                )
    return rows

def collect_re_selection(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(root / "results_combined_re_selection.csv"):
        rows.append(
            {
                "group": row["group"],
                "group_name": row["group_name"],
                "selected_key": row["selected_key"],
                "selected_re": row["selected_re"],
            }
        )
    return rows

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
        "\\centering\\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    if body_rows:
        lines.extend(" & ".join(row) + " \\\\" for row in body_rows)
    else:
        lines.append(
            f"\\multicolumn{{{len(headers)}}}{{c}}{{No significant effects.}} \\\\"
        )
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

def split_contrast(contrast: str) -> tuple[str, str]:
    left, right = contrast.split(" - ")
    return left, right

def build_dominance_counts(
    pairwise_rows: list[dict[str, str | float]], scopes: list[str]
) -> dict[str, dict[str, dict[str, dict[str, dict[str, int]]]]]:
    counts = {
        scope: {
            subset: {
                group: {arch: {"wins": 0, "losses": 0} for arch in ARCH_ORDER}
                for group in GROUP_ORDER
            }
            for subset in SUBSET_LABELS
        }
        for scope in scopes
    }
    for row in pairwise_rows:
        left_arch, right_arch = split_contrast(row["contrast"])
        if row["estimate"] > 0:
            winner = left_arch
            loser = right_arch
        else:
            winner = right_arch
            loser = left_arch
        counts[row["scope"]][row["subset"]][row["group"]][winner]["wins"] += 1
        counts[row["scope"]][row["subset"]][row["group"]][loser]["losses"] += 1
    return counts

def build_entity_matrix(
    entity_rows: list[dict[str, str | float]], scopes: list[str]
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    matrix = {
        scope: {
            subset: {arch: {group: "" for group in GROUP_ORDER} for arch in ARCH_ORDER}
            for subset in SUBSET_LABELS
        }
        for scope in scopes
    }
    for row in entity_rows:
        matrix[row["scope"]][row["subset"]][row["arch"]][row["group"]] = format_entity_cell(
            row["estimate"]
        )
    return matrix

def write_generalization_gap_tables(
    output_dir: Path,
    combined_anova: list[dict[str, str | float]],
    combined_pairwise: list[dict[str, str | float]],
    combined_entity_by_subset: list[dict[str, str | float]],
    combined_entity: list[dict[str, str | float]],
    per_split_pairwise: list[dict[str, str | float]],
    per_split_entity: list[dict[str, str | float]],
    re_rows: list[dict[str, str]],
) -> None:
    anova_body = [
        [
            row["group_name"],
            row["effect_label"],
            f"{row['f_value']:.2f}",
            format_p(row["p_value"]),
        ]
        for row in sorted(
            combined_anova,
            key=lambda row: (
                GROUP_ORDER.index(row["group"]),
                ANOVA_EFFECT_ORDER.index(row["effect_key"]),
            ),
        )
    ]
    write_table(
        output_dir / "stat_gap_significant_anova.tex",
        "@{}llrr@{}",
        [
            "\\textbf{Group}",
            "\\textbf{Effect}",
            "$F$",
            "$p$",
        ],
        anova_body,
        "Significant generalization-gap ANOVA summary from a joint analysis of Split~1 and Split~2, with Split included as a factor.",
        "tab:gap_significant_anova",
    )

    combined_counts = build_dominance_counts(combined_pairwise, ["combined"])
    combined_dominance_body = []
    for subset in ("train", "test"):
        for arch in ARCH_ORDER:
            total_wins = 0
            total_losses = 0
            group_cells = []
            for group in GROUP_ORDER:
                wins = combined_counts["combined"][subset][group][arch]["wins"]
                losses = combined_counts["combined"][subset][group][arch]["losses"]
                total_wins += wins
                total_losses += losses
                group_cells.append(format_count_cell(wins, losses))
            combined_dominance_body.append(
                [
                    SUBSET_LABELS[subset],
                    latex_escape(arch),
                    *group_cells,
                    format_count_cell(total_wins, total_losses),
                ]
            )
    write_table(
        output_dir / "stat_gap_arch_dominance.tex",
        "@{}llccccc@{}",
        [
            "\\textbf{Subset}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
            "\\textbf{Total}",
        ],
        combined_dominance_body,
        "Generalization-gap architecture summary pooled across Split~1 and Split~2. For each architecture, the Word, Sentence, Complex, and Basic cells report the numbers of statistically significant pairwise wins and losses within that category and subset; the Total column aggregates across the four categories.",
        "tab:gap_arch_dominance",
    )

    combined_entity_subset_body = []
    for group in GROUP_ORDER:
        group_rows = {
            row["subset"]: row
            for row in combined_entity_by_subset
            if row["group"] == group
        }
        train_row = group_rows["train"]
        test_row = group_rows["test"]
        combined_entity_subset_body.append(
            [
                GROUP_LABELS[group],
                format_entity_cell(train_row["estimate"]),
                format_p(train_row["p_value"]),
                format_entity_cell(test_row["estimate"]),
                format_p(test_row["p_value"]),
            ]
        )
    write_table(
        output_dir / "stat_gap_entity_by_subset.tex",
        "@{}lrrrr@{}",
        [
            "\\textbf{Group}",
            "\\textbf{Train est.}",
            "\\textbf{Train $p$}",
            "\\textbf{Test est.}",
            "\\textbf{Test $p$}",
        ],
        combined_entity_subset_body,
        "Entity effects by train/test subset after pooling Split~1 and Split~2. Positive \\texttt{Ent} entries mean entity vectors help; \\texttt{NoEnt} means the no-entity condition is favored. This table includes both significant and non-significant subset-level effects.",
        "tab:gap_entity_by_subset",
    )

    combined_entity_matrix = build_entity_matrix(combined_entity, ["combined"])
    combined_entity_body = []
    for subset in ("train", "test"):
        for arch in ARCH_ORDER:
            combined_entity_body.append(
                [
                    SUBSET_LABELS[subset],
                    latex_escape(arch),
                    combined_entity_matrix["combined"][subset][arch]["word_group"] or "--",
                    combined_entity_matrix["combined"][subset][arch]["sentence_group"] or "--",
                    combined_entity_matrix["combined"][subset][arch]["complex_event"] or "--",
                    combined_entity_matrix["combined"][subset][arch]["basic_event"] or "--",
                ]
            )
    write_table(
        output_dir / "stat_gap_entity_matrix.tex",
        "@{}llcccc@{}",
        [
            "\\textbf{Subset}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
        ],
        combined_entity_body,
        "Entity-effect matrix after pooling Split~1 and Split~2. Cells show the favored condition and absolute estimate magnitude; \\texttt{--} indicates no significant effect.",
        "tab:gap_entity_matrix",
    )

    per_split_counts = build_dominance_counts(per_split_pairwise, SPLIT_ORDER)
    per_split_dominance_body = []
    for split in SPLIT_ORDER:
        for subset in ("train", "test"):
            for arch in ARCH_ORDER:
                total_wins = 0
                total_losses = 0
                group_cells = []
                for group in GROUP_ORDER:
                    wins = per_split_counts[split][subset][group][arch]["wins"]
                    losses = per_split_counts[split][subset][group][arch]["losses"]
                    total_wins += wins
                    total_losses += losses
                    group_cells.append(format_count_cell(wins, losses))
                per_split_dominance_body.append(
                    [
                        split,
                        SUBSET_LABELS[subset],
                        latex_escape(arch),
                        *group_cells,
                        format_count_cell(total_wins, total_losses),
                    ]
                )
    write_table(
        output_dir / "stat_gap_per_split_arch_dominance.tex",
        "@{}lllccccc@{}",
        [
            "\\textbf{Split}",
            "\\textbf{Subset}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
            "\\textbf{Total}",
        ],
        per_split_dominance_body,
        "Generalization-gap architecture summary shown separately for Split~1 and Split~2. For each architecture, the Word, Sentence, Complex, and Basic cells report the numbers of statistically significant pairwise wins and losses within that category, split, and subset; the Total column aggregates across the four categories.",
        "tab:gap_per_split_arch_dominance",
    )

    per_split_entity_matrix = build_entity_matrix(per_split_entity, SPLIT_ORDER)
    per_split_entity_body = []
    for split in SPLIT_ORDER:
        for subset in ("train", "test"):
            for arch in ARCH_ORDER:
                per_split_entity_body.append(
                    [
                        split,
                        SUBSET_LABELS[subset],
                        latex_escape(arch),
                        per_split_entity_matrix[split][subset][arch]["word_group"] or "--",
                        per_split_entity_matrix[split][subset][arch]["sentence_group"] or "--",
                        per_split_entity_matrix[split][subset][arch]["complex_event"] or "--",
                        per_split_entity_matrix[split][subset][arch]["basic_event"] or "--",
                    ]
                )
    write_table(
        output_dir / "stat_gap_per_split_entity_matrix.tex",
        "@{}lllcccc@{}",
        [
            "\\textbf{Split}",
            "\\textbf{Subset}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
        ],
        per_split_entity_body,
        "Entity-effect matrix shown separately for Split~1 and Split~2. Cells show the favored condition and absolute estimate magnitude; \\texttt{--} indicates no significant effect.",
        "tab:gap_per_split_entity_matrix",
    )

    re_body = [
        [
            row["group_name"],
            latex_escape(row["selected_key"]),
            latex_escape(row["selected_re"]),
        ]
        for row in sorted(re_rows, key=lambda row: GROUP_ORDER.index(row["group"]))
    ]
    write_table(
        output_dir / "stat_gap_re_selection.tex",
        "@{}lll@{}",
        [
            "\\textbf{Group}",
            "\\textbf{Selected RE}",
            "\\textbf{Structure}",
        ],
        re_body,
        "Compact combined-model random-effects selection summary for the generalization-gap results.",
        "tab:gap_re_selection",
    )

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_anova = collect_combined_anova(args.input_root)
    combined_pairwise = collect_combined_arch_pairwise(args.input_root)
    combined_entity_by_subset = collect_combined_entity_by_subset(args.input_root)
    combined_entity = collect_combined_entity_by_arch(args.input_root)
    per_split_pairwise = collect_per_split_arch_pairwise(args.input_root)
    per_split_entity = collect_per_split_entity_by_arch(args.input_root)
    re_rows = collect_re_selection(args.input_root)

    write_generalization_gap_tables(
        args.output_dir,
        combined_anova,
        combined_pairwise,
        combined_entity_by_subset,
        combined_entity,
        per_split_pairwise,
        per_split_entity,
        re_rows,
    )

    print(f"Wrote compact generalization-gap summaries to {args.output_dir}")

if __name__ == "__main__":
    main()
