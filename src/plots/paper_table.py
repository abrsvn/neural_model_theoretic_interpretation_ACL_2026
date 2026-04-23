"""Paper-table regeneration."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from plots.summary_validation import require_unambiguous_summary_rows

GROUP_ORDER = (
    ("Word", "Word"),
    ("Sentence", "Sent"),
    ("Complex_Event", "Cmplx"),
    ("Basic_Event", "Basic"),
)
ARCH_ORDER = (
    ("SIMPLE_RN", "SRN"),
    ("SIMPLE_GRU", "GRU"),
    ("SIMPLE_LSTM", "LSTM"),
    ("ABS_ATTN", "Attn AbsPE"),
    ("ROPE_ATTN", "Attn RoPE"),
)
ENTITY_ORDER = (
    ("no_entity", "$-$ent"),
    ("with_entity", "$+$ent"),
)


def write_main_results_table(
    summary_rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    experiment_id: str | None = None,
) -> None:
    """Write the paper's main test-results table from per-run summary rows."""

    if experiment_id is None:
        require_unambiguous_summary_rows(summary_rows)
        filtered_rows = summary_rows
    else:
        filtered_rows = [
            row for row in summary_rows if row["experiment_id"] == experiment_id
        ]
    if not filtered_rows:
        if experiment_id is None:
            raise ValueError("No summary rows provided")
        raise ValueError(f"No summary rows found for experiment_id={experiment_id}")
    arch_entries = _ordered_arch_entries(filtered_rows)

    advantages: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    model_ids: set[tuple[str, int, int]] = set()

    for row in filtered_rows:
        key = (str(row["model_type"]), str(row["entity_condition"]), str(row["group_name"]))
        advantage = float(row["avg_test_advantage"])
        advantages[key].append(advantage)
        model_ids.add(
            (
                str(row["model_type"]),
                int(row["split"]),
                int(row["model_index"]),
            )
        )

    arch_model_counts = [
        len(
            {
                (model_type, split, model_index)
                for model_type, split, model_index in model_ids
                if model_type == arch
            }
        )
        for arch, _ in arch_entries
    ]
    count_values = {count for count in arch_model_counts if count > 0}
    if len(count_values) == 1:
        average_text = (
            f"Averaged over {count_values.pop()} models per architecture/entity condition. "
        )
    else:
        average_text = "Averaged over the available models per architecture/entity condition. "

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"& & \textbf{Word} & \textbf{Sent} & \textbf{Cmplx} & \textbf{Basic} \\",
        r"\midrule",
    ]

    for arch_index, (model_type, arch_label) in enumerate(arch_entries):
        for entity_index, (entity_condition, entity_label) in enumerate(ENTITY_ORDER):
            if entity_index == 0:
                row_cells = [rf"\multirow{{2}}{{*}}{{{arch_label}}}", entity_label]
            else:
                row_cells = ["", entity_label]

            for group_name, _ in GROUP_ORDER:
                values = advantages.get((model_type, entity_condition, group_name), [])
                if values:
                    row_cells.append(f"{_mean(values):.2f}")
                else:
                    row_cells.append("--")

            lines.append(" & ".join(row_cells) + r" \\")

        if arch_index < len(arch_entries) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Test-only advantage scores (described $-$ competing) by architecture, test group and $\pm$entity. "
                + average_text
                + r"Higher is better.}"
            ),
            r"\label{tab:main_results_model}",
            r"\end{table}",
        ]
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _ordered_arch_entries(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    observed_model_types = {str(row["model_type"]) for row in rows}
    arch_entries = [
        (model_type, arch_label)
        for model_type, arch_label in ARCH_ORDER
        if model_type in observed_model_types
    ]
    if not arch_entries:
        raise ValueError("No recognized model types found in summary rows")
    return arch_entries
