"""Plotting for cross-model appendix distribution figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy import stats

from .metadata import (
    COMPETING_EVENT_COLORS,
    COMPETING_EVENT_ORDER,
    GROUP_DISPLAY,
    GROUP_KEYS,
    SPLITS,
    arch_display,
    sort_architectures,
)


logger = logging.getLogger(__name__)

ENTITIES = ("noent", "ent")

_TAB10 = plt.cm.tab10.colors
_TAB10_LIGHT = [
    tuple(min(1.0, channel + 0.3) for channel in color[:3])
    for color in _TAB10
]
_GAP_COLORS = {
    ("S1", "train"): "#a6cee3",
    ("S1", "test"): "#1f78b4",
    ("S2", "train"): "#fdbf6f",
    ("S2", "test"): "#ff7f00",
}
_LR_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_advantage_histograms(
    data: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    archs: list[str],
    output_path: str | Path,
) -> None:
    """Plot one histogram panel per group/architecture with +/-entity overlays."""

    present_groups = [group for group in GROUP_KEYS if group in data]
    active_archs = _archs_with_data(data, archs)
    colors_ent, colors_noent = _architecture_colors(active_archs)

    figure, axes = plt.subplots(
        len(present_groups),
        len(active_archs),
        figsize=(4.5 * len(active_archs), 3.5 * len(present_groups)),
        squeeze=False,
    )

    for row_index, group in enumerate(present_groups):
        for column_index, arch in enumerate(active_archs):
            axis = axes[row_index, column_index]
            has_data = False

            for entity, alpha, color_map, annotation_x, annotation_alignment in (
                ("noent", 0.4, colors_noent, 0.03, "left"),
                ("ent", 0.6, colors_ent, 0.97, "right"),
            ):
                pooled = pool_arch_entity(data, group, arch, entity)
                if pooled is None:
                    continue
                has_data = True

                entity_label = "+ent" if entity == "ent" else "-ent"
                color = color_map.get(arch, "#888888")
                axis.hist(
                    pooled,
                    bins=30,
                    alpha=alpha,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    label=f"{entity_label} (n={len(pooled)})",
                )
                bin_width = ((pooled.max() - pooled.min()) / 30 if pooled.max() > pooled.min() else 1)
                _add_kde_overlay(
                    axis,
                    pooled,
                    color=color,
                    n_total=len(pooled),
                    bin_width=bin_width,
                )
                _annotate_distribution(
                    axis,
                    pooled,
                    x=annotation_x,
                    y=0.95 if entity == "ent" else 0.60,
                    ha=annotation_alignment,
                )

            if not has_data:
                axis.set_visible(False)
                continue

            axis.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            axis.legend(fontsize=6, loc="upper center")
            axis.grid(axis="y", alpha=0.3)

            if row_index == 0:
                axis.set_title(arch_display(arch), fontsize=10, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{GROUP_DISPLAY[group]}\nCount", fontsize=9)
            if row_index == len(present_groups) - 1:
                axis.set_xlabel("Advantage", fontsize=8)

    figure.suptitle(
        "Per-Sentence Advantage Distributions (+/- Entity Vectors)",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_advantage_boxplots(
    data: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    archs: list[str],
    output_path: str | Path,
) -> None:
    """Plot boxplots with +/-entity grouped side by side for each architecture."""

    present_groups = [group for group in GROUP_KEYS if group in data]
    active_archs = _archs_with_data(data, archs)
    colors_ent, colors_noent = _architecture_colors(active_archs)

    figure, axes = plt.subplots(
        len(present_groups),
        1,
        figsize=(max(14, 3 * len(active_archs)), 4.5 * len(present_groups)),
        squeeze=False,
    )

    for group_index, group in enumerate(present_groups):
        axis = axes[group_index, 0]
        box_data: list[np.ndarray] = []
        box_positions: list[float] = []
        box_colors: list[str] = []
        tick_positions: list[float] = []
        tick_labels: list[str] = []

        for arch_index, arch in enumerate(active_archs):
            center = arch_index * 3
            for entity, offset in (("noent", -0.45), ("ent", 0.45)):
                pooled = pool_arch_entity(data, group, arch, entity)
                if pooled is None:
                    continue
                box_data.append(pooled)
                box_positions.append(center + offset)
                box_colors.append(
                    colors_ent.get(arch, "#555555")
                    if entity == "ent"
                    else colors_noent.get(arch, "#999999")
                )
            tick_positions.append(center)
            tick_labels.append(arch_display(arch))

        if not box_data:
            axis.set_visible(False)
            continue

        boxplot = axis.boxplot(
            box_data,
            positions=box_positions,
            widths=0.7,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )

        for patch, color in zip(boxplot["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for pooled, position in zip(box_data, box_positions, strict=True):
            quartile_1, median, quartile_3 = np.percentile(pooled, [25, 50, 75])
            text = (
                f"med={median:.2f}\n"
                f"IQR=[{quartile_1:.2f},{quartile_3:.2f}]\n"
                f"range=[{pooled.min():.2f},{pooled.max():.2f}]\n"
                f"n={len(pooled)}"
            )
            axis.annotate(
                text,
                xy=(position, quartile_3 + 1.5 * (quartile_3 - quartile_1)),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=5,
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "lightyellow",
                    "alpha": 0.8,
                    "edgecolor": "gray",
                },
            )

        axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, fontsize=9)
        axis.set_ylabel(f"{GROUP_DISPLAY[group]}\nAdvantage Score", fontsize=9)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=0.4, label="-ent"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#555555", alpha=0.6, label="+ent"),
            ],
            fontsize=8,
            loc="lower right",
        )

    figure.suptitle(
        "Per-Sentence Advantage Score Distributions (+/- Entity Vectors)",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_gap_by_split(
    sentence_data: pd.DataFrame,
    archs: list[str],
    output_path: str | Path,
) -> None:
    """Plot grouped bar charts of mean advantage by split and train/test phase."""

    present_groups = [group for group in GROUP_KEYS if group in sentence_data["group"].values]
    active_archs = sort_architectures(
        [arch for arch in archs if arch in sentence_data["arch"].values]
    )

    n_columns = 2
    n_rows = max((len(present_groups) + 1) // 2, 1)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(14, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    bar_width = 0.18
    combo_order = [(split, phase) for split in SPLITS for phase in ("train", "test")]

    for axis_index, group in enumerate(present_groups):
        axis = axes_flat[axis_index]
        group_rows = sentence_data[sentence_data["group"] == group]
        x_positions = np.arange(len(active_archs))

        for bar_index, (split, phase) in enumerate(combo_order):
            subset = group_rows[
                (group_rows["split"] == split) & (group_rows["train_or_test"] == phase)
            ]
            if subset.empty:
                continue

            means: list[float] = []
            sems: list[float] = []
            for arch in active_archs:
                arch_subset = subset[subset["arch"] == arch]
                model_means = arch_subset.groupby("model_id")["advantage"].mean()
                means.append(float(model_means.mean()))
                sems.append(
                    float(model_means.std(ddof=1) / np.sqrt(len(model_means)))
                    if len(model_means) > 1
                    else float("nan")
                )

            axis.bar(
                x_positions + (bar_index - 1.5) * bar_width,
                means,
                bar_width,
                yerr=sems,
                color=_GAP_COLORS[(split, phase)],
                edgecolor="black",
                linewidth=0.4,
                label=f"{split} {phase}",
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels([arch_display(arch) for arch in active_archs], fontsize=9, rotation=15)
        axis.set_ylabel("Mean Advantage", fontsize=10)
        axis.set_title(GROUP_DISPLAY[group], fontsize=12, fontweight="bold")
        axis.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.6)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=7, ncol=2, loc="lower right")

    for axis_index in range(len(present_groups), len(axes_flat)):
        axes_flat[axis_index].set_visible(False)

    figure.suptitle("Generalization Gap by Split and Architecture", fontsize=14, fontweight="bold")
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_competitor_histograms(
    counts: dict[str, dict[str, list[int]]],
    output_path: str | Path,
    *,
    pool_train_test: bool = True,
) -> None:
    """Plot competing-event histograms for the four systematicity categories."""

    if pool_train_test:
        rows = [
            (f"Split {split[-1]}", pool_competing_counts(counts, split))
            for split in SPLITS
        ]
    else:
        rows = [
            (key.replace("_", " ").title(), counts[key])
            for key in _COMPETING_SPLIT_PHASE_KEYS
            if key in counts
        ]

    present_categories = [
        category
        for category in COMPETING_EVENT_ORDER
        if any(category in row_data for _, row_data in rows)
    ]
    figure, axes = plt.subplots(
        len(rows),
        len(present_categories),
        figsize=(4 * len(present_categories), 4 * len(rows)),
        squeeze=False,
    )

    for row_index, (row_label, row_data) in enumerate(rows):
        for column_index, category in enumerate(present_categories):
            axis = axes[row_index, column_index]
            if category not in row_data or not row_data[category]:
                axis.set_visible(False)
                continue

            values = np.array(row_data[category], dtype=float)
            max_value = int(values.max())
            bins = np.arange(-0.5, max_value + 1.5, 1)
            color = COMPETING_EVENT_COLORS[category]

            axis.hist(
                values,
                bins=bins,
                color=color,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
            )
            _add_kde_overlay(
                axis,
                values,
                color=color,
                n_total=len(values),
                bin_width=1.0,
                bw_method=0.4,
            )
            _annotate_competing_hist(axis, values)
            axis.set_xticks(range(0, max_value + 1))
            axis.grid(axis="y", alpha=0.3)

            if row_index == len(rows) - 1:
                axis.set_xlabel("Number of Competitors", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\nCount", fontsize=10)
            if row_index == 0:
                axis.set_title(category, fontsize=11, fontweight="bold")

    suffix = "by Split" if pool_train_test else "by Split (Train / Test)"
    figure.suptitle(
        f"Competing Events per Sentence {suffix}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_competitor_boxplots(
    counts: dict[str, dict[str, list[int]]],
    output_path: str | Path,
    *,
    pool_train_test: bool = True,
) -> None:
    """Plot competing-event boxplots for the four systematicity categories."""

    if pool_train_test:
        panels = [
            (f"Split {split[-1]}", pool_competing_counts(counts, split))
            for split in SPLITS
        ]
    else:
        panels = [
            (key.replace("_", " ").title(), counts[key])
            for key in _COMPETING_SPLIT_PHASE_KEYS
            if key in counts
        ]

    n_columns = min(len(panels), 2)
    n_rows = (len(panels) + n_columns - 1) // n_columns
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7 * n_columns, 6 * n_rows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis_index, (panel_title, panel_data) in enumerate(panels):
        axis = axes_flat[axis_index]
        present_categories = [
            category for category in COMPETING_EVENT_ORDER if category in panel_data
        ]
        values_list: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        positions: list[int] = []

        for category_index, category in enumerate(present_categories, start=1):
            values = np.array(panel_data[category], dtype=float)
            values_list.append(values)
            labels.append(category)
            colors.append(COMPETING_EVENT_COLORS[category])
            positions.append(category_index)

        if not values_list:
            axis.set_visible(False)
            continue

        boxplot = axis.boxplot(
            values_list,
            positions=positions,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        )

        for patch, color in zip(boxplot["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        rng = np.random.default_rng(42)
        for position, values, color in zip(positions, values_list, colors, strict=True):
            jitter = rng.uniform(-0.15, 0.15, size=len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                color=color,
                s=10,
                alpha=0.4,
                zorder=5,
                edgecolors="black",
                linewidths=0.2,
            )

        for position, values in zip(positions, values_list, strict=True):
            quartile_1, median, quartile_3 = np.percentile(values, [25, 50, 75])
            text = (
                f"med={median:.0f}, IQR=[{quartile_1:.0f},{quartile_3:.0f}]\n"
                f"range=[{values.min():.0f},{values.max():.0f}], n={len(values)}"
            )
            axis.annotate(
                text,
                xy=(position, values.min()),
                xytext=(0, -12),
                textcoords="offset points",
                fontsize=6,
                ha="center",
                va="top",
            )

        axis.set_title(panel_title, fontsize=12, fontweight="bold")
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)
        if axis_index % n_columns == 0:
            axis.set_ylabel("Number of Competitors", fontsize=10)

    for axis_index in range(len(panels), len(axes_flat)):
        axes_flat[axis_index].set_visible(False)

    suffix = "" if pool_train_test else " (Train / Test)"
    figure.suptitle(
        f"Competing Events per Sentence (Box Plots){suffix}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_lr_grid(
    trajectories: dict[str, list[dict[str, object]]],
    output_path: str | Path,
) -> None:
    """Plot mean learning-rate schedules with SE shading by architecture."""

    ordered = sort_architectures(list(trajectories))
    n_panels = len(ordered)
    n_columns = 2
    n_rows = max((n_panels + 1) // 2, 1)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for axis_index, arch_name in enumerate(ordered):
        axis = axes_flat[axis_index]
        display_name = arch_display(arch_name)
        color = _architecture_line_color(arch_name, ordered)
        lr_lists = [trajectory["lr"] for trajectory in trajectories[arch_name]]
        lengths = [len(lr_values) for lr_values in lr_lists]
        min_length = min(lengths)
        lr_matrix = np.array([lr_values[:min_length] for lr_values in lr_lists], dtype=float)
        n_models = lr_matrix.shape[0]
        mean_lr = lr_matrix.mean(axis=0)
        epochs = np.arange(1, min_length + 1)

        if n_models > 1:
            se_lr = lr_matrix.std(axis=0, ddof=1) / np.sqrt(n_models)
            axis.plot(epochs, mean_lr, color=color, linewidth=2, label=f"Mean +/- SE (N={n_models})")
            axis.fill_between(epochs, mean_lr - se_lr, mean_lr + se_lr, color=color, alpha=0.3)
        else:
            axis.plot(epochs, mean_lr, color=color, linewidth=2, label=f"N={n_models}")

        axis.set_yscale("log")
        axis.set_title(display_name, fontsize=12, fontweight="bold")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    for axis_index in range(n_panels, len(axes_flat)):
        axes_flat[axis_index].set_visible(False)

    figure.supxlabel("Epoch", fontsize=12)
    figure.supylabel("Learning Rate", fontsize=12)
    figure.suptitle(
        "Adaptive Learning Rate Schedules by Architecture",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_and_close(figure, output_path)


def plot_best_epoch_boxplot(
    trajectories: dict[str, list[dict[str, object]]],
    output_path: str | Path,
) -> None:
    """Plot the best-epoch distribution by architecture."""

    ordered = sort_architectures(list(trajectories))
    labels: list[str] = []
    data: list[list[int]] = []
    colors: list[str] = []
    for arch_name in ordered:
        labels.append(arch_display(arch_name))
        colors.append(_architecture_line_color(arch_name, ordered))
        best_epochs: list[int] = []
        for trajectory in trajectories[arch_name]:
            val_scores = trajectory["val_score"]
            best_epochs.append(_find_first_best_epoch(val_scores) + 1)
        data.append(best_epochs)

    figure, axis = plt.subplots(figsize=(8, 5))
    boxplot = axis.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
    )

    for patch, color in zip(boxplot["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    for position, epochs_list, color in zip(range(1, len(data) + 1), data, colors, strict=True):
        jitter = rng.uniform(-0.15, 0.15, size=len(epochs_list))
        axis.scatter(
            np.full(len(epochs_list), position) + jitter,
            epochs_list,
            color=color,
            s=15,
            alpha=0.6,
            zorder=5,
            edgecolors="black",
            linewidths=0.3,
        )

    all_lengths = {len(trajectory["val_score"]) for entries in trajectories.values() for trajectory in entries}
    max_epochs = max(all_lengths)
    axis.set_ylabel("Epoch of Best Model Save", fontsize=11)
    axis.set_title("Distribution of Best-Model Epochs by Architecture", fontsize=13, fontweight="bold")
    axis.axhline(
        y=max_epochs,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Max epochs ({max_epochs})",
    )
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    _save_and_close(figure, output_path)


def pool_arch_entity(
    data: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    group: str,
    arch: str,
    entity: str,
) -> np.ndarray | None:
    """Pool all per-sentence advantages for one `(arch, entity)` group cell."""

    key = (arch, entity)
    if group not in data or key not in data[group]:
        return None
    arrays = data[group][key]
    if not arrays:
        return None
    return np.concatenate(arrays)


def _architecture_colors(archs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    colors_ent: dict[str, str] = {}
    colors_noent: dict[str, str] = {}
    for index, arch in enumerate(archs):
        color_index = index % len(_TAB10)
        colors_ent[arch] = "#%02x%02x%02x" % tuple(
            int(channel * 255) for channel in _TAB10[color_index][:3]
        )
        colors_noent[arch] = "#%02x%02x%02x" % tuple(
            int(min(255, channel * 255)) for channel in _TAB10_LIGHT[color_index][:3]
        )
    return (colors_ent, colors_noent)


def _archs_with_data(
    data: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    archs: list[str],
) -> list[str]:
    return [
        arch
        for arch in archs
        if any((arch, entity) in group_data for group_data in data.values() for entity in ENTITIES)
    ]


def _annotate_distribution(
    axis: Axes,
    values: np.ndarray,
    *,
    x: float,
    y: float,
    ha: str,
) -> None:
    text = (
        f"n={len(values)}\n"
        f"mean={values.mean():.3f}\n"
        f"SD={values.std(ddof=1):.3f}"
    )
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        fontsize=6,
        va="top",
        ha=ha,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "gray",
        },
    )


def _annotate_competing_hist(axis: Axes, values: np.ndarray) -> None:
    sd = f"SD={values.std(ddof=1):.2f}" if len(values) > 1 else "SD=n/a"
    text = f"n={len(values)}\nmean={values.mean():.2f}\n{sd}"
    axis.text(
        0.97,
        0.95,
        text,
        transform=axis.transAxes,
        fontsize=7,
        va="top",
        ha="right",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "gray",
        },
    )


def _add_kde_overlay(
    axis: Axes,
    values: np.ndarray,
    *,
    color: str,
    n_total: int,
    bin_width: float,
    bw_method: float | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3 or values.std() == 0:
        return
    kde = stats.gaussian_kde(values, bw_method=bw_method)
    x_grid = np.linspace(values.min() - 0.5, values.max() + 0.5, 300)
    kde_y = kde(x_grid) * n_total * bin_width
    axis.plot(x_grid, kde_y, color=color, linewidth=1.5, alpha=0.8)


def pool_competing_counts(
    counts: dict[str, dict[str, list[int]]],
    split: str,
) -> dict[str, list[int]]:
    pooled: dict[str, list[int]] = {category: [] for category in COMPETING_EVENT_ORDER}
    for phase in ("train", "test"):
        key = f"{split}_{phase}"
        if key not in counts:
            continue
        for category in COMPETING_EVENT_ORDER:
            pooled[category].extend(counts[key].get(category, []))
    return {category: values for category, values in pooled.items() if values}


_COMPETING_SPLIT_PHASE_KEYS = tuple(
    f"{split}_{phase}" for split in SPLITS for phase in ("train", "test")
)


def _architecture_line_color(arch_name: str, ordered: list[str]) -> str:
    if arch_name not in ordered:
        return _LR_COLORS[0]
    return _LR_COLORS[ordered.index(arch_name) % len(_LR_COLORS)]


def _find_first_best_epoch(val_score: list[float]) -> int:
    running_max = float("-inf")
    best_epoch = 0
    for epoch_index, score in enumerate(val_score):
        if score > running_max:
            running_max = score
            best_epoch = epoch_index
    return best_epoch


def _save_and_close(figure: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"Wrote {path}")
