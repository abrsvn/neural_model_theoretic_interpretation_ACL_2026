"""Paper-figure regeneration."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from data.dataset import SYSTEMATICITY_GROUPS
from plots.summary_validation import (
    require_unambiguous_history_records,
    require_unambiguous_summary_rows,
)


MODEL_ORDER = {
    "SIMPLE_RN": 0,
    "SIMPLE_GRU": 1,
    "SIMPLE_LSTM": 2,
    "ABS_ATTN": 3,
    "ROPE_ATTN": 4,
}

GROUP_COLORS = {
    "Word": "#2A6F97",
    "Sentence": "#4D908E",
    "Complex_Event": "#F4A261",
    "Basic_Event": "#BC4749",
}


def plot_entity_vector_comparison(
    summary_rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    experiment_id: str | None = None,
    show_title: bool = True,
) -> None:
    """Plot the paper's entity-vector comparison with described/competing panels."""

    filtered_rows = _filter_summary_rows(summary_rows, experiment_id=experiment_id)
    if not filtered_rows:
        raise ValueError("No summary rows available for entity-vector comparison plotting")

    model_keys = _ordered_model_keys(filtered_rows)
    fig, axes = plt.subplots(
        len(model_keys),
        2,
        figsize=(16, 4 * len(model_keys)),
        squeeze=False,
    )

    x = np.arange(len(SYSTEMATICITY_GROUPS))
    width = 0.35

    for row_index, model_key in enumerate(model_keys):
        model_rows = [
            row for row in filtered_rows if row["model_type"] == model_key
        ]
        paper_label = _paper_label_for_model(model_rows, model_key)
        with_entity_rows = [
            row for row in model_rows if row["entity_condition"] == "with_entity"
        ]
        no_entity_rows = [
            row for row in model_rows if row["entity_condition"] == "no_entity"
        ]
        if not with_entity_rows or not no_entity_rows:
            raise ValueError(
                f"Entity-vector comparison requires both entity conditions for {model_key}"
            )

        ax_desc = axes[row_index, 0]
        ax_comp = axes[row_index, 1]

        with_desc, with_desc_err = _aggregate_metric(with_entity_rows, "avg_test_described")
        no_desc, no_desc_err = _aggregate_metric(no_entity_rows, "avg_test_described")
        with_comp, with_comp_err = _aggregate_metric(with_entity_rows, "avg_test_competing")
        no_comp, no_comp_err = _aggregate_metric(no_entity_rows, "avg_test_competing")
        with_theoretical_min, _ = _aggregate_metric(with_entity_rows, "avg_theoretical_min")
        no_theoretical_min, _ = _aggregate_metric(no_entity_rows, "avg_theoretical_min")

        bars_with_desc = ax_desc.bar(
            x - width / 2,
            with_desc,
            width=width,
            yerr=with_desc_err,
            capsize=3,
            color="#111111",
            label="With entity vectors",
        )
        bars_no_desc = ax_desc.bar(
            x + width / 2,
            no_desc,
            width=width,
            yerr=no_desc_err,
            capsize=3,
            color="white",
            edgecolor="#111111",
            hatch="///",
            label="Without entity vectors",
        )

        for bar, value in zip(bars_with_desc, with_desc):
            if np.isnan(value):
                continue
            ax_desc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar, value in zip(bars_no_desc, no_desc):
            if np.isnan(value):
                continue
            ax_desc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax_desc.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
        ax_desc.set_ylabel("Comprehension score")
        ax_desc.set_title(f"{paper_label} - Described events")
        ax_desc.set_xticks(x)
        ax_desc.set_xticklabels([group.replace("_", " ") for group in SYSTEMATICITY_GROUPS])
        ax_desc.set_ylim(0.0, 1.2)
        ax_desc.grid(axis="y", alpha=0.25)
        ax_desc.legend(loc="upper right")
        _add_yaxis_theoretical_maximum_marker(ax_desc, fontsize=8, text_y=1.03)

        bars_with_comp = ax_comp.bar(
            x - width / 2,
            with_comp,
            width=width,
            yerr=with_comp_err,
            capsize=3,
            color="#111111",
            label="With entity vectors",
        )
        bars_no_comp = ax_comp.bar(
            x + width / 2,
            no_comp,
            width=width,
            yerr=no_comp_err,
            capsize=3,
            color="white",
            edgecolor="#111111",
            hatch="///",
            label="Without entity vectors",
        )

        for bar, value in zip(bars_with_comp, with_comp):
            if np.isnan(value):
                continue
            y_pos = bar.get_height() - 0.02 if bar.get_height() < 0 else bar.get_height() + 0.015
            va = "top" if bar.get_height() < 0 else "bottom"
            ax_comp.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )
        for bar, value in zip(bars_no_comp, no_comp):
            if np.isnan(value):
                continue
            y_pos = bar.get_height() - 0.02 if bar.get_height() < 0 else bar.get_height() + 0.015
            va = "top" if bar.get_height() < 0 else "bottom"
            ax_comp.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )

        for index, theoretical_min in enumerate(with_theoretical_min):
            if np.isnan(theoretical_min):
                continue
            _add_theoretical_bound_marker(
                ax_comp,
                x[index] - width / 2,
                theoretical_min,
                width * 0.8,
                text_offset=-0.03,
                fontsize=8,
            )
        for index, theoretical_min in enumerate(no_theoretical_min):
            if np.isnan(theoretical_min):
                continue
            _add_theoretical_bound_marker(
                ax_comp,
                x[index] + width / 2,
                theoretical_min,
                width * 0.8,
                text_offset=-0.03,
                fontsize=8,
            )

        ax_comp.axhline(0.0, color="#6B7280", linestyle="--", linewidth=1)
        ax_comp.set_ylabel("Comprehension score")
        ax_comp.set_title(f"{paper_label} - Competing events")
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels([group.replace("_", " ") for group in SYSTEMATICITY_GROUPS])
        ax_comp.set_ylim(-1.2, 0.05)
        ax_comp.grid(axis="y", alpha=0.25)
        ax_comp.legend(loc="lower right")

    if show_title:
        title = "Entity-vector comparison"
        if experiment_id is not None:
            title = f"{title}: {experiment_id}"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()
    _save_figure(fig, output_path)


def plot_generalization_gap(
    summary_rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
) -> list[Path]:
    """Create one train-vs-test generalization-gap figure per model/entity combination."""

    filtered_rows = _filter_summary_rows(summary_rows, experiment_id=experiment_id)
    if not filtered_rows:
        raise ValueError("No summary rows available for generalization-gap plotting")

    grouped_rows = defaultdict(list)
    for row in filtered_rows:
        grouped_rows[(row["model_type"], row["entity_condition"])].append(row)

    output_paths: list[Path] = []
    for model_type, entity_condition in sorted(
        grouped_rows,
        key=lambda item: (MODEL_ORDER.get(item[0], 999), item[1]),
    ):
        rows = grouped_rows[(model_type, entity_condition)]
        paper_label = _paper_label_for_model(rows, model_type)
        fig, (ax_desc, ax_comp) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        x = np.arange(len(SYSTEMATICITY_GROUPS))
        width = 0.35

        train_desc, train_desc_err = _aggregate_metric(rows, "avg_train_described")
        test_desc, test_desc_err = _aggregate_metric(rows, "avg_test_described")
        train_comp, train_comp_err = _aggregate_metric(rows, "avg_train_competing")
        test_comp, test_comp_err = _aggregate_metric(rows, "avg_test_competing")
        test_theoretical_min, _ = _aggregate_metric(rows, "avg_theoretical_min")
        train_theoretical_min, _ = _aggregate_metric(rows, "avg_train_theoretical_min")

        ax_desc.bar(
            x - width / 2,
            train_desc,
            width=width,
            yerr=train_desc_err,
            capsize=3,
            color="#1F2937",
            label="Train",
        )
        ax_desc.bar(
            x + width / 2,
            test_desc,
            width=width,
            yerr=test_desc_err,
            capsize=3,
            color="white",
            edgecolor="#111827",
            hatch="//",
            label="Test",
        )
        ax_desc.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
        ax_desc.set_title("Described events")
        ax_desc.set_ylabel("Comprehension score")
        ax_desc.grid(axis="y", alpha=0.25)
        _add_yaxis_theoretical_maximum_marker(ax_desc, fontsize=8, text_y=1.03)

        ax_comp.bar(
            x - width / 2,
            train_comp,
            width=width,
            yerr=train_comp_err,
            capsize=3,
            color="#1F2937",
            label="Train",
        )
        ax_comp.bar(
            x + width / 2,
            test_comp,
            width=width,
            yerr=test_comp_err,
            capsize=3,
            color="white",
            edgecolor="#111827",
            hatch="//",
            label="Test",
        )
        for index, theoretical_min in enumerate(train_theoretical_min):
            if not np.isnan(theoretical_min):
                _add_theoretical_bound_marker(
                    ax_comp,
                    x[index] - width / 2,
                    theoretical_min,
                    width * 0.8,
                    text_offset=-0.03,
                    fontsize=8,
                )
        for index, theoretical_min in enumerate(test_theoretical_min):
            if not np.isnan(theoretical_min):
                _add_theoretical_bound_marker(
                    ax_comp,
                    x[index] + width / 2,
                    theoretical_min,
                    width * 0.8,
                    text_offset=-0.03,
                    fontsize=8,
                )
        ax_comp.set_title("Competing events")
        ax_comp.grid(axis="y", alpha=0.25)

        for axis in (ax_desc, ax_comp):
            axis.set_xticks(x)
            axis.set_xticklabels([group.replace("_", " ") for group in SYSTEMATICITY_GROUPS])
        ax_desc.legend(loc="upper right")

        fig.suptitle(f"Generalization gap: {paper_label} ({_format_entity_title(entity_condition)})")
        fig.tight_layout()

        filename = (
            f"aggregated_comparison_{_plot_stem(model_type)}_{entity_condition}_extended.png"
        )
        output_path = Path(output_dir) / filename
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    return output_paths


def _aggregate_metric(
    rows: list[dict[str, object]],
    metric_name: str,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    errors: list[float] = []
    for group_name in SYSTEMATICITY_GROUPS:
        values = [
            float(row[metric_name])
            for row in rows
            if row["group_name"] == group_name
        ]
        mean_value, err_value, _ = _mean_and_stderr(values)
        means.append(mean_value)
        errors.append(err_value)
    return (means, errors)


def _add_yaxis_theoretical_maximum_marker(
    axis: Axes,
    *,
    fontsize: int,
    text_y: float,
) -> None:
    x_min, x_max = axis.get_xlim()
    x_pos = x_min + (x_max - x_min) * 0.01
    axis.plot([x_pos], [1.0], "o", color="#8B0000", markersize=8, zorder=15, clip_on=False)
    axis.text(
        x_pos,
        text_y,
        "1.00\n(max possible)",
        ha="center",
        va="bottom",
        fontsize=fontsize,
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


def _filter_summary_rows(
    summary_rows: list[dict[str, object]],
    *,
    experiment_id: str | None,
) -> list[dict[str, object]]:
    if experiment_id is None:
        require_unambiguous_summary_rows(summary_rows)
        return summary_rows
    return [row for row in summary_rows if row["experiment_id"] == experiment_id]


def _ordered_model_keys(rows: list[dict[str, object]]) -> list[str]:
    return sorted(
        {str(row["model_type"]) for row in rows},
        key=lambda model_type: MODEL_ORDER.get(model_type, 999),
    )


def _paper_label_for_model(rows: list[dict[str, object]], model_type: str) -> str:
    for row in rows:
        if row["model_type"] == model_type:
            return str(row["paper_label"])
    raise ValueError(f"Could not find paper label for model type {model_type}")


def _format_entity_title(entity_condition: str) -> str:
    if entity_condition == "with_entity":
        return "With entity vectors"
    if entity_condition == "no_entity":
        return "Without entity vectors"
    raise ValueError(f"Unknown entity condition: {entity_condition}")


def _mean_and_stderr(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return (float("nan"), float("nan"), 0)

    array = np.asarray(values, dtype=float)
    mean_value = float(array.mean())
    if array.shape[0] == 1:
        return (mean_value, float("nan"), 1)
    stderr = float(array.std(ddof=1) / np.sqrt(array.shape[0]))
    return (mean_value, stderr, int(array.shape[0]))


def _stderr_array(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return np.full(values.shape[1], float("nan"), dtype=float)
    return values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug


def _plot_stem(model_type: str) -> str:
    if model_type == "SIMPLE_RN":
        return "srn"
    if model_type == "SIMPLE_LSTM":
        return "lstm"
    if model_type == "SIMPLE_GRU":
        return "gru"
    if model_type == "ABS_ATTN":
        return "attn_sinusoidal"
    if model_type == "ROPE_ATTN":
        return "attn_rope"
    return _slugify(model_type)


def plot_training_curves(
    history_records: list[dict[str, object]],
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
) -> list[Path]:
    """Create grouped training-curve plots by model/entity condition."""

    filtered_records = _filter_history_records(
        history_records,
        experiment_id=experiment_id,
    )
    if not filtered_records:
        raise ValueError("No history records available for training-curve plotting")

    grouped_records = defaultdict(list)
    for record in filtered_records:
        grouped_records[(record["model_type"], record["entity_condition"])].append(record)

    output_paths: list[Path] = []
    for model_type, entity_condition in sorted(
        grouped_records,
        key=lambda item: (MODEL_ORDER.get(item[0], 999), item[1]),
    ):
        records = grouped_records[(model_type, entity_condition)]
        paper_label = _paper_label_for_model(records, model_type)

        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        grid = fig.add_gridspec(2, 2)
        ax_loss = fig.add_subplot(grid[0, 0])
        ax_score = fig.add_subplot(grid[0, 1])
        ax_tests = fig.add_subplot(grid[1, :])

        _plot_history_metric(ax_loss, records, "train_loss", label="Train loss", color="#1D4ED8")
        _plot_history_metric(ax_loss, records, "val_loss", label="Validation loss", color="#B91C1C")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend(loc="upper right")

        _plot_history_metric(ax_score, records, "train_score", label="Train score", color="#1D4ED8")
        _plot_history_metric(ax_score, records, "val_score", label="Validation score", color="#B91C1C")
        ax_score.set_title("Train and validation scores")
        ax_score.set_xlabel("Epoch")
        ax_score.set_ylabel("Score")
        ax_score.grid(alpha=0.25)
        ax_score.legend(loc="upper right")

        for group_name in SYSTEMATICITY_GROUPS:
            _plot_history_metric(
                ax_tests,
                records,
                f"test_{group_name}",
                label=group_name.replace("_", " "),
                color=GROUP_COLORS[group_name],
            )
        ax_tests.set_title("Test group scores")
        ax_tests.set_xlabel("Epoch")
        ax_tests.set_ylabel("Score")
        ax_tests.grid(alpha=0.25)
        ax_tests.legend(loc="upper right", ncol=2)

        fig.suptitle(f"Training curves: {paper_label} ({_format_entity_title(entity_condition)})")

        filename = f"training_curves_{_plot_stem(model_type)}_{entity_condition}.png"
        output_path = Path(output_dir) / filename
        _save_figure(fig, output_path)
        output_paths.append(output_path)

    return output_paths


def _plot_history_metric(
    axis: Axes,
    history_records: list[dict[str, object]],
    metric_name: str,
    *,
    label: str,
    color: str,
) -> None:
    epochs, means, errors = _aggregate_history_metric(history_records, metric_name)
    if epochs is None:
        return

    axis.plot(epochs, means, label=label, color=color, linewidth=2)
    axis.fill_between(
        epochs,
        means - errors,
        means + errors,
        color=color,
        alpha=0.2,
    )


def _aggregate_history_metric(
    history_records: list[dict[str, object]],
    metric_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    curves = []
    for record in history_records:
        history = record["history"]
        if metric_name not in history or not history[metric_name]:
            continue
        curves.append(np.asarray(history[metric_name], dtype=float))

    if not curves:
        return (None, None, None)

    min_length = min(len(curve) for curve in curves)
    truncated = np.vstack([curve[:min_length] for curve in curves])
    means = truncated.mean(axis=0)
    errors = _stderr_array(truncated)
    epochs = np.arange(1, min_length + 1)
    return (epochs, means, errors)


def _filter_history_records(
    history_records: list[dict[str, object]],
    *,
    experiment_id: str | None,
) -> list[dict[str, object]]:
    if experiment_id is None:
        require_unambiguous_history_records(history_records)
        return history_records
    return [record for record in history_records if record["experiment_id"] == experiment_id]
