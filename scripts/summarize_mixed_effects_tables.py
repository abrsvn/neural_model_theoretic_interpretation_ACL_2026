#!/usr/bin/env python3
"""Generate compact mixed-effects appendix summaries from result CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "statistical_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "compact_mixed_effects_summaries"
PHASE_LABELS = {"test": "Test", "train": "Train"}
ARCH_ORDER = [
    "SRN",
    "GRU",
    "LSTM",
    "Attn_AbsPE",
    "Attn_RoPE",
    "Attn_AbsPE_H80",
    "Attn_RoPE_H80",
]
ARCH_FAMILY = {
    "SRN": "recurrent",
    "LSTM": "recurrent",
    "GRU": "recurrent",
    "Attn_AbsPE": "attention",
    "Attn_RoPE": "attention",
    "Attn_AbsPE_H80": "attention",
    "Attn_RoPE_H80": "attention",
}
GROUP_LABELS = {
    "word_group": "Word",
    "sentence_group": "Sentence",
    "complex_event": "Complex Event",
    "basic_event": "Basic Event",
}
GROUP_ORDER = ["word_group", "sentence_group", "complex_event", "basic_event"]
ANOVA_EFFECT_LABELS = {
    "arch": "Arch",
    "entity": "Entity",
    "split": "Split",
    "arch:entity": "Arch $\\times$ Ent",
    "arch:split": "Arch $\\times$ Split",
    "entity:split": "Entity $\\times$ Split",
    "arch:entity:split": "Arch $\\times$ Ent $\\times$ Split",
}
SIGNIFICANCE_THRESHOLD = 0.05
BASIC_EVENT_CONTRAST_ORDER = [
    "LSTM - GRU",
    "SRN - Attn_AbsPE",
    "SRN - Attn_RoPE",
    "SRN - Attn_AbsPE_H80",
    "SRN - Attn_RoPE_H80",
    "LSTM - Attn_AbsPE",
    "LSTM - Attn_RoPE",
    "LSTM - Attn_AbsPE_H80",
    "LSTM - Attn_RoPE_H80",
    "GRU - Attn_AbsPE",
    "GRU - Attn_RoPE",
    "GRU - Attn_AbsPE_H80",
    "GRU - Attn_RoPE_H80",
]
ARCH_DISPLAY = {
    "Attn_AbsPE": "AbsPE",
    "Attn_RoPE": "RoPE",
    "Attn_AbsPE_H80": "AbsPE H80",
    "Attn_RoPE_H80": "RoPE H80",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing the mixed-effects result CSVs.",
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

def format_signed(value: float) -> str:
    return f"{value:+.3f}"

def format_cell(value: str) -> str:
    return value if value else "--"

def format_entity_cell(estimate: float) -> str:
    favored_condition = "Ent" if estimate < 0 else "NoEnt"
    return f"{favored_condition} {abs(estimate):.3f}"

def collect_significant_anova(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for phase in ("test", "train"):
        for group_stem in GROUP_ORDER:
            anova_path = root / f"results_{phase}_{group_stem}_anova.csv"
            for row in read_csv_rows(anova_path):
                effect_key = row[""]
                p_value = parse_float(row["Pr(>F)"])
                if p_value >= SIGNIFICANCE_THRESHOLD:
                    continue
                rows.append(
                    {
                        "phase": phase,
                        "group": group_stem,
                        "group_name": GROUP_LABELS[group_stem],
                        "effect_key": effect_key,
                        "effect_label": ANOVA_EFFECT_LABELS[effect_key],
                        "f_value": parse_float(row["F value"]),
                        "p_value": p_value,
                    }
                )
    return rows

def collect_significant_arch_pairwise(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for phase in ("test", "train"):
        for group_stem in GROUP_ORDER:
            pairwise_path = root / f"results_{phase}_{group_stem}_arch_pairwise.csv"
            for row in read_csv_rows(pairwise_path):
                p_value = parse_float(row["p.value"])
                if p_value >= SIGNIFICANCE_THRESHOLD:
                    continue
                rows.append(
                    {
                        "phase": phase,
                        "group": group_stem,
                        "group_name": GROUP_LABELS[group_stem],
                        "contrast": row["contrast"],
                        "estimate": parse_float(row["estimate"]),
                        "p_value": p_value,
                    }
                )
    return rows

def collect_significant_entity_effect(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for phase in ("test", "train"):
        for group_stem in GROUP_ORDER:
            entity_path = root / f"results_{phase}_{group_stem}_entity_effect.csv"
            for row in read_csv_rows(entity_path):
                p_value = parse_float(row["p.value"])
                if p_value >= SIGNIFICANCE_THRESHOLD:
                    continue
                rows.append(
                    {
                        "phase": phase,
                        "group": group_stem,
                        "group_name": GROUP_LABELS[group_stem],
                        "arch": row["arch"],
                        "estimate": parse_float(row["estimate"]),
                        "p_value": p_value,
                    }
                )
    return rows

def collect_test_entity_main_effects(root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for group_stem in GROUP_ORDER:
        anova_rows = read_csv_rows(root / f"results_test_{group_stem}_anova.csv")
        entity_row = next(row for row in anova_rows if row[""] == "entity")
        rows.append(
            {
                "group_name": GROUP_LABELS[group_stem],
                "num_df": parse_float(entity_row["NumDF"]),
                "den_df": parse_float(entity_row["DenDF"]),
                "f_value": parse_float(entity_row["F value"]),
                "p_value": parse_float(entity_row["Pr(>F)"]),
            }
        )
    return rows

def collect_basic_event_arch_by_entity(
    root: Path,
) -> dict[tuple[str, str], tuple[float, float]]:
    rows = read_csv_rows(root / "results_test_basic_event_arch_by_entity.csv")
    contrasts: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        contrast = row["contrast"]
        entity = row["entity"]
        estimate = parse_float(row["estimate"])
        p_value = parse_float(row["p.value"])
        contrasts[(contrast, entity)] = (estimate, p_value)
        contrasts[(reverse_contrast(contrast), entity)] = (-estimate, p_value)
    return contrasts

def write_table(
    path: Path,
    column_spec: str,
    headers: list[str],
    body_rows: list[list[str]],
    caption: str,
    label: str,
    size_command: str = "\\small",
) -> None:
    lines = [
        "\\begin{table}[t]",
        f"\\centering{size_command}",
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

def reverse_contrast(contrast: str) -> str:
    left, right = split_contrast(contrast)
    return f"{right} - {left}"

def format_arch_contrast(contrast: str) -> str:
    left, right = split_contrast(contrast)
    return f"{ARCH_DISPLAY.get(left, left)} - {ARCH_DISPLAY.get(right, right)}"

def format_arch_label(arch: str) -> str:
    return ARCH_DISPLAY.get(arch, arch)

def build_dominance_counts(
    pairwise_rows: list[dict[str, str | float]],
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        phase: {
            group: {arch: {"wins": 0, "losses": 0} for arch in ARCH_ORDER}
            for group in GROUP_ORDER
        }
        for phase in PHASE_LABELS
    }
    for row in pairwise_rows:
        left_arch, right_arch = split_contrast(row["contrast"])
        if row["estimate"] > 0:
            winner = left_arch
            loser = right_arch
        else:
            winner = right_arch
            loser = left_arch
        counts[row["phase"]][row["group"]][winner]["wins"] += 1
        counts[row["phase"]][row["group"]][loser]["losses"] += 1
    return counts

def collect_within_family_pairwise(
    pairwise_rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for row in pairwise_rows:
        left_arch, right_arch = split_contrast(row["contrast"])
        if ARCH_FAMILY[left_arch] != ARCH_FAMILY[right_arch]:
            continue
        rows.append(row)
    return rows

def build_entity_matrix(
    entity_rows: list[dict[str, str | float]],
) -> dict[str, dict[str, dict[str, str]]]:
    matrix = {
        phase: {arch: {group: "" for group in GROUP_ORDER} for arch in ARCH_ORDER}
        for phase in PHASE_LABELS
    }
    for row in entity_rows:
        matrix[row["phase"]][row["arch"]][row["group"]] = format_entity_cell(row["estimate"])
    return matrix

def write_mixed_effects_tables(
    output_dir: Path,
    anova_rows: list[dict[str, str | float]],
    pairwise_rows: list[dict[str, str | float]],
    entity_rows: list[dict[str, str | float]],
    test_entity_main_effects: list[dict[str, str | float]],
    basic_event_arch_by_entity: dict[tuple[str, str], tuple[float, float]],
) -> None:
    anova_body = [
        [
            PHASE_LABELS[row["phase"]],
            row["group_name"],
            row["effect_label"],
            f"{row['f_value']:.2f}",
            format_p(row["p_value"]),
        ]
        for row in anova_rows
    ]
    write_table(
        output_dir / "stat_mixed_effects_significant_anova.tex",
        "@{}lllrr@{}",
        [
            "\\textbf{Phase}",
            "\\textbf{Group}",
            "\\textbf{Effect}",
            "$F$",
            "$p$",
        ],
        anova_body,
        "Significant ANOVA summary for the mixed-effects results.",
        "tab:mixed_effects_significant_anova",
    )

    dominance_counts = build_dominance_counts(pairwise_rows)
    dominance_body = []
    for phase in ("test", "train"):
        for arch in ARCH_ORDER:
            total_wins = 0
            total_losses = 0
            group_cells = []
            for group in GROUP_ORDER:
                wins = dominance_counts[phase][group][arch]["wins"]
                losses = dominance_counts[phase][group][arch]["losses"]
                total_wins += wins
                total_losses += losses
                group_cells.append(f"{wins}-{losses}")
            dominance_body.append(
                [
                    PHASE_LABELS[phase],
                    latex_escape(format_arch_label(arch)),
                    *group_cells,
                    f"{total_wins}-{total_losses}",
                ]
            )
    write_table(
        output_dir / "stat_mixed_effects_arch_dominance.tex",
        "@{}llccccc@{}",
        [
            "\\textbf{Phase}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
            "\\textbf{Total}",
        ],
        dominance_body,
        "Architecture-dominance summary for the mixed-effects results. Each cell reports the number of statistically significant pairwise wins and losses within that group.",
        "tab:mixed_effects_arch_dominance",
        size_command="\\scriptsize",
    )

    within_family_rows = collect_within_family_pairwise(pairwise_rows)
    exceptions_body = [
        [
            PHASE_LABELS[row["phase"]],
            row["group_name"],
            latex_escape(format_arch_contrast(row["contrast"])),
            format_signed(row["estimate"]),
            format_p(row["p_value"]),
        ]
        for row in within_family_rows
    ]
    write_table(
        output_dir / "stat_mixed_effects_arch_family_exceptions.tex",
        "@{}lllrr@{}",
        [
            "\\textbf{Phase}",
            "\\textbf{Group}",
            "\\textbf{Contrast}",
            "\\textbf{Estimate}",
            "$p$",
        ],
        exceptions_body,
        "Within-family architecture contrasts for the mixed-effects results. These rows capture the recurrent-vs-recurrent and attention-vs-attention exceptions hidden by the dominance summary.",
        "tab:mixed_effects_arch_family_exceptions",
        size_command="\\scriptsize",
    )

    basic_event_body = []
    for contrast in BASIC_EVENT_CONTRAST_ORDER:
        noent_estimate, noent_p = basic_event_arch_by_entity[(contrast, "noent")]
        ent_estimate, ent_p = basic_event_arch_by_entity[(contrast, "ent")]
        basic_event_body.append(
            [
                latex_escape(format_arch_contrast(contrast)),
                format_signed(noent_estimate),
                format_p(noent_p),
                format_signed(ent_estimate),
                format_p(ent_p),
            ]
        )
    write_table(
        output_dir / "stat_mixed_effects_basic_event_arch_by_entity.tex",
        "@{}lrrrr@{}",
        [
            "\\textbf{Contrast}",
            "\\textbf{$-$ent est.}",
            "\\textbf{$-$ent $p$}",
            "\\textbf{$+$ent est.}",
            "\\textbf{$+$ent $p$}",
        ],
        basic_event_body,
        "Targeted Basic architecture contrasts by entity condition for the mixed-effects results. Positive estimates favor the architecture on the left of the contrast. These are the key pairwise comparisons cited in Observation~3.",
        "tab:basic_event_arch_by_entity",
        size_command="\\scriptsize",
    )

    entity_matrix = build_entity_matrix(entity_rows)
    entity_body = []
    for phase in ("test", "train"):
        for arch in ARCH_ORDER:
            entity_body.append(
                [
                    PHASE_LABELS[phase],
                    latex_escape(format_arch_label(arch)),
                    format_cell(entity_matrix[phase][arch]["word_group"]),
                    format_cell(entity_matrix[phase][arch]["sentence_group"]),
                    format_cell(entity_matrix[phase][arch]["complex_event"]),
                    format_cell(entity_matrix[phase][arch]["basic_event"]),
                ]
            )
    write_table(
        output_dir / "stat_mixed_effects_entity_matrix.tex",
        "@{}llcccc@{}",
        [
            "\\textbf{Phase}",
            "\\textbf{Architecture}",
            "\\textbf{Word}",
            "\\textbf{Sentence}",
            "\\textbf{Complex}",
            "\\textbf{Basic}",
        ],
        entity_body,
        "Entity-effect matrix for the mixed-effects results. Cells show the favored condition and absolute estimate magnitude; \\texttt{--} indicates no significant effect.",
        "tab:mixed_effects_entity_matrix",
        size_command="\\scriptsize",
    )

    test_entity_main_effect_body = [
        [
            row["group_name"],
            f"{row['f_value']:.2f}",
            f"{int(row['num_df'])}, {row['den_df']:.1f}",
            format_p(row["p_value"]),
        ]
        for row in test_entity_main_effects
    ]
    write_table(
        output_dir / "stat_mixed_effects_test_entity_main_effects.tex",
        "@{}lrrr@{}",
        [
            "\\textbf{Group}",
            "$F$",
            "\\textbf{df}",
            "$p$",
        ],
        test_entity_main_effect_body,
        "Test-only entity main effects from the mixed-effects ANOVA models. This table includes both significant and non-significant entity effects so the null results cited in Observation~4 are directly visible.",
        "tab:test_entity_main_effects",
    )

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anova = collect_significant_anova(args.input_root)
    pairwise = collect_significant_arch_pairwise(args.input_root)
    entity = collect_significant_entity_effect(args.input_root)
    test_entity_main_effects = collect_test_entity_main_effects(args.input_root)
    basic_event_arch_by_entity = collect_basic_event_arch_by_entity(args.input_root)

    write_mixed_effects_tables(
        args.output_dir,
        anova,
        pairwise,
        entity,
        test_entity_main_effects,
        basic_event_arch_by_entity,
    )
    print(f"Wrote compact mixed-effects summaries to {args.output_dir}")

if __name__ == "__main__":
    main()
