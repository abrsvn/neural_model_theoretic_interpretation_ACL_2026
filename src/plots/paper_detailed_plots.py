"""Detailed appendix plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from data.dataset import SYSTEMATICITY_GROUPS


MODEL_ORDER = {
    "SIMPLE_RN": 0,
    "SIMPLE_GRU": 1,
    "SIMPLE_LSTM": 2,
    "ABS_ATTN": 3,
    "ROPE_ATTN": 4,
}

MODEL_LABELS = {
    "SIMPLE_RN": "SRN",
    "SIMPLE_LSTM": "LSTM",
    "SIMPLE_GRU": "GRU",
    "ABS_ATTN": "Attn AbsPE",
    "ROPE_ATTN": "Attn RoPE",
}

ENTITY_ORDER = ("no_entity", "with_entity")
ENTITY_LABELS = {
    "no_entity": "No Entity",
    "with_entity": "With Entity",
}


def plot_entity_vector_comparison_detailed(
    detail_rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    experiment_id: str | None = None,
) -> None:
    """Create the appendix detailed entity-vector comparison figure."""

    if experiment_id is None:
        _require_unambiguous_detail_rows(detail_rows)

    filtered_rows = [
        row
        for row in detail_rows
        if row["split"] == "test"
        and (experiment_id is None or row["experiment_id"] == experiment_id)
    ]
    if not filtered_rows:
        raise ValueError("No detailed rows available for entity-vector detailed plotting")

    model_types = sorted(
        {str(row["model_type"]) for row in filtered_rows},
        key=lambda model_type: MODEL_ORDER.get(model_type, 999),
    )
    if not model_types:
        raise ValueError("No recognized model types found for detailed entity plotting")

    fig, axes = plt.subplots(
        len(model_types),
        len(SYSTEMATICITY_GROUPS),
        figsize=(20, 4 * len(model_types)),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.95, bottom=0.05)
    fig.suptitle(
        "Entity Vector Comparison: Compositional Systematicity Breakdown",
        fontsize=16,
        fontweight="bold",
    )

    for row_index, model_type in enumerate(model_types):
        for column_index, group_name in enumerate(SYSTEMATICITY_GROUPS):
            axis = axes[row_index, column_index]
            entity_data = _aggregate_detailed_entity_rows(
                filtered_rows,
                model_type=model_type,
                group_name=group_name,
            )
            if not entity_data:
                axis.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                axis.set_title(
                    f"{_model_label(model_type)}\n{group_name.replace('_', ' ')}"
                )
                axis.axis("off")
                continue

            all_levels = _ordered_complexity_levels(entity_data)
            entity_conditions = [
                entity_condition
                for entity_condition in ENTITY_ORDER
                if entity_condition in entity_data
            ]

            bar_width = 0.15
            spacing = 0.05
            group_gap = 0.3
            current_x = 0.0

            for entity_index, entity_condition in enumerate(entity_conditions):
                if entity_index > 0:
                    current_x += group_gap

                for level_index, level_name in enumerate(all_levels):
                    color, hatch = _gradient_color_and_hatch(
                        level_index,
                        len(all_levels),
                    )
                    described_score, competing_score = _level_scores(
                        entity_data[entity_condition],
                        level_name,
                    )
                    x_desc = current_x
                    x_comp = current_x + bar_width

                    axis.bar(
                        x_desc,
                        described_score,
                        bar_width,
                        color=color,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    axis.bar(
                        x_comp,
                        competing_score,
                        bar_width,
                        color=color,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    _add_bar_labels(axis, [x_desc], [described_score], fontsize=5)
                    _add_bar_labels(axis, [x_comp], [competing_score], fontsize=5)
                    current_x += 2 * bar_width + spacing

            marker_width = bar_width * 1.2
            current_x_bounds = 0.0
            for entity_index, entity_condition in enumerate(entity_conditions):
                if entity_index > 0:
                    current_x_bounds += group_gap

                bounds_by_level = {
                    level_name: (theoretical_max, theoretical_min)
                    for level_name, _, _, _, theoretical_max, theoretical_min in entity_data[
                        entity_condition
                    ]
                }
                for level_name in all_levels:
                    if level_name in bounds_by_level:
                        _, competing_min = bounds_by_level[level_name]
                        _add_theoretical_bound_marker(
                            axis,
                            current_x_bounds + bar_width,
                            competing_min,
                            marker_width,
                            text_offset=-0.05,
                            fontsize=5,
                        )
                    current_x_bounds += 2 * bar_width + spacing

            axis.set_ylabel("Comprehension Score", fontsize=9)
            axis.set_title(
                f"{_model_label(model_type)}\n{group_name.replace('_', ' ')}",
                fontsize=10,
                fontweight="bold",
            )
            axis.grid(axis="y", alpha=0.3, linestyle="--")
            axis.axhline(y=0.0, color="black", linestyle="-", linewidth=0.8)
            axis.set_ylim(-1.2, 1.35)
            axis.set_xlim(-0.2, current_x)
            _add_yaxis_theoretical_maximum_marker(axis)

            if entity_conditions:
                n_bars_per_group = len(all_levels) * 2
                total_width_per_group = (
                    n_bars_per_group * bar_width + (len(all_levels) - 1) * spacing
                )
                group_centers = []
                x_start = 0.0
                for entity_index in range(len(entity_conditions)):
                    if entity_index > 0:
                        x_start += group_gap
                    group_centers.append(x_start + total_width_per_group / 2)
                    x_start += total_width_per_group
                axis.set_xticks(group_centers)
                axis.set_xticklabels(
                    [ENTITY_LABELS[entity_condition] for entity_condition in entity_conditions],
                    fontsize=9,
                )

            _add_compositional_legend(axis, all_levels, entity_data, fontsize=7)

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _require_unambiguous_detail_rows(detail_rows: list[dict[str, object]]) -> None:
    """
    Ambiguity Guard:
    Prevents silent merging of experiments that share model_type identifiers (e.g. ABS_ATTN)
    but differ in capacity (hidden_dim) or other architectural properties.
    When a user passes CSV files from multiple experiments without filtering via --experiment-id,
    this guard prevents generating corrupted, averaged plots or tables.
    """
    signatures_by_key: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in detail_rows:
        key = (
            str(row["model_type"]),
            str(row["entity_condition"]),
            str(row["group_name"]),
            str(row["complexity_level"]),
            str(row["split"]),
        )
        signatures_by_key[key].add(str(row["experiment_id"]))

    problems = []
    for key in sorted(signatures_by_key):
        experiment_ids = signatures_by_key[key]
        if len(experiment_ids) < 2:
            continue
        problems.append(
            f"{key[0]} / {key[1]} / {key[2]} / {key[3]} / {key[4]} -> "
            + ", ".join(sorted(experiment_ids))
        )

    if problems:
        raise ValueError(
            "Mixed experiment rows require --experiment-id. Ambiguous detail rows:\n"
            + "\n".join(problems)
        )


def _aggregate_detailed_entity_rows(
    detail_rows: list[dict[str, object]],
    *,
    model_type: str,
    group_name: str,
) -> dict[str, list[tuple[str, float, float, int, float, float]]]:
    run_level_rows: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        if row["model_type"] != model_type or row["group_name"] != group_name:
            continue
        run_level_rows[
            (
                str(row["entity_condition"]),
                int(row["run_split"]),
                int(row["model_index"]),
                str(row["complexity_level"]),
            )
        ].append(row)

    aggregated: dict[str, dict[str, list[tuple[int, float, float, int, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (entity_condition, run_split, _model_index, level_name), rows in run_level_rows.items():
        described_scores = [float(row["described_score"]) for row in rows]
        competing_scores = [float(row["competing_score"]) for row in rows]
        theoretical_max_values = _nonempty_float_values(rows, "theoretical_max")
        theoretical_min_values = _nonempty_float_values(rows, "theoretical_min")
        aggregated[entity_condition][level_name].append(
            (
                run_split,
                float(np.mean(described_scores)),
                float(np.mean(competing_scores)),
                len(rows),
                float(np.mean(theoretical_max_values)) if theoretical_max_values else float("nan"),
                float(np.mean(theoretical_min_values)) if theoretical_min_values else float("nan"),
            )
        )

    averaged: dict[str, list[tuple[str, float, float, int, float, float]]] = {}
    for entity_condition, levels_data in aggregated.items():
        averaged_levels = []
        for level_name, values in levels_data.items():
            averaged_levels.append(
                (
                    level_name,
                    float(np.mean([described for _, described, _, _, _, _ in values])),
                    float(np.mean([competing for _, _, competing, _, _, _ in values])),
                    _aggregate_unique_level_count(
                        values,
                        model_type=model_type,
                        group_name=group_name,
                        entity_condition=entity_condition,
                        level_name=level_name,
                    ),
                    float(np.mean([theoretical_max for _, _, _, _, theoretical_max, _ in values])),
                    float(np.mean([theoretical_min for _, _, _, _, _, theoretical_min in values])),
                )
            )
        averaged[entity_condition] = averaged_levels
    return averaged


def _aggregate_unique_level_count(
    values: list[tuple[int, float, float, int, float, float]],
    *,
    model_type: str,
    group_name: str,
    entity_condition: str,
    level_name: str,
) -> int:
    counts_by_split: dict[int, set[int]] = defaultdict(set)
    for run_split, _, _, count, _, _ in values:
        counts_by_split[run_split].add(count)

    inconsistent_counts = [
        f"s{run_split}={sorted(counts)}"
        for run_split, counts in sorted(counts_by_split.items())
        if len(counts) > 1
    ]
    if inconsistent_counts:
        raise ValueError(
            "Inconsistent detailed row counts for "
            f"{model_type} / {group_name} / {entity_condition} / {level_name}: "
            + ", ".join(inconsistent_counts)
        )

    return sum(
        next(iter(counts))
        for _, counts in sorted(counts_by_split.items())
    )


def _nonempty_float_values(
    rows: list[dict[str, object]],
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row[field_name]
        if value in ("", None):
            continue
        values.append(float(value))
    return values


def _ordered_complexity_levels(
    entity_data: dict[str, list[tuple[str, float, float, int, float, float]]],
) -> list[str]:
    all_levels = set()
    for levels_data in entity_data.values():
        for level_name, _, _, _, _, _ in levels_data:
            all_levels.add(level_name)

    def sort_key(level_name: str) -> tuple[int, str]:
        if level_name == "Canonical":
            return (0, level_name)
        modifier_count = 2 if "+" in level_name else 1
        return (modifier_count, level_name)

    return sorted(all_levels, key=sort_key)


def _level_scores(
    levels_data: list[tuple[str, float, float, int, float, float]],
    level_name: str,
) -> tuple[float, float]:
    for candidate_level, described_score, competing_score, _, _, _ in levels_data:
        if candidate_level == level_name:
            return (described_score, competing_score)
    return (float("nan"), float("nan"))


def _add_bar_labels(
    axis: Axes,
    positions: list[float],
    scores: list[float],
    *,
    fontsize: int,
) -> None:
    for position, score in zip(positions, scores, strict=True):
        if np.isnan(score):
            continue
        y_pos = score + 0.02 if score >= 0 else score - 0.02
        vertical_alignment = "bottom" if score >= 0 else "top"
        axis.text(
            position,
            y_pos,
            f"{score:.2f}",
            ha="center",
            va=vertical_alignment,
            fontsize=fontsize,
        )


def _gradient_color_and_hatch(index: int, total: int) -> tuple[str, str | None]:
    if total == 1:
        return ("#000000", None)
    value = int(255 * index / (total - 1))
    color = f"#{value:02x}{value:02x}{value:02x}"
    if index == 0:
        return (color, None)
    if index == total - 1:
        return (color, "xxx")
    if index < total / 2:
        return (color, "///")
    return (color, "\\\\\\")


def _add_yaxis_theoretical_maximum_marker(axis: Axes) -> None:
    x_min, x_max = axis.get_xlim()
    x_pos = x_min + (x_max - x_min) * 0.01
    axis.plot([x_pos], [1.0], "o", color="#8B0000", markersize=8, zorder=15, clip_on=False)
    axis.text(
        x_pos,
        1.05,
        "1.00\n(max possible)",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#8B0000",
        fontweight="bold",
        zorder=15,
    )


def _add_theoretical_bound_marker(
    axis: Axes,
    x_pos: float,
    value: float,
    marker_width: float,
    *,
    text_offset: float,
    fontsize: int,
) -> None:
    axis.plot(
        [x_pos - marker_width / 2, x_pos + marker_width / 2],
        [value, value],
        color="#8B0000",
        linewidth=2.5,
        alpha=0.9,
        linestyle="-",
        zorder=10,
    )
    axis.text(
        x_pos,
        value + text_offset,
        f"{value:.2f}",
        ha="center",
        va="top",
        fontsize=fontsize,
        color="#8B0000",
        fontweight="bold",
        zorder=11,
    )


def _add_compositional_legend(
    axis: Axes,
    all_levels: list[str],
    entity_data: dict[str, list[tuple[str, float, float, int, float, float]]],
    *,
    fontsize: int,
) -> None:
    handles = []
    labels = []
    for index, level_name in enumerate(all_levels):
        level_count = _legend_level_count(entity_data, level_name)
        if level_count is None:
            continue
        color, hatch = _gradient_color_and_hatch(index, len(all_levels))
        handles.append(
            plt.matplotlib.patches.Patch(
                facecolor=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
        )
        labels.append(f"{level_name} (n={level_count})")
    axis.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=fontsize,
        framealpha=0.7,
        edgecolor="black",
        fancybox=False,
    )


def _legend_level_count(
    entity_data: dict[str, list[tuple[str, float, float, int, float, float]]],
    level_name: str,
) -> int | None:
    counts_by_entity: dict[str, set[int]] = defaultdict(set)
    for entity_condition, levels_data in entity_data.items():
        for candidate_level, _, _, count, _, _ in levels_data:
            if candidate_level == level_name:
                counts_by_entity[entity_condition].add(count)

    if not counts_by_entity:
        return None

    inconsistent_counts = [
        f"{entity_condition}={sorted(counts)}"
        for entity_condition, counts in sorted(counts_by_entity.items())
        if len(counts) > 1
    ]
    if inconsistent_counts:
        raise ValueError(
            f"Inconsistent legend counts for level {level_name}: "
            + ", ".join(inconsistent_counts)
        )

    unique_counts = {
        next(iter(counts))
        for counts in counts_by_entity.values()
    }
    if len(unique_counts) > 1:
        raise ValueError(
            f"Entity conditions disagree on legend count for level {level_name}: "
            + ", ".join(
                f"{entity_condition}={next(iter(counts))}"
                for entity_condition, counts in sorted(counts_by_entity.items())
            )
        )

    return next(iter(unique_counts))


def _model_label(model_type: str) -> str:
    return MODEL_LABELS.get(model_type, model_type)
