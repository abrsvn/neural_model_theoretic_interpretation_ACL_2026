"""Competing-event and training-trajectory appendix outputs."""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metadata import COMPETING_EVENT_ORDER, make_architecture_label, parse_checkpoint_name
from .plotting import (
    plot_best_epoch_boxplot,
    plot_competitor_boxplots,
    plot_competitor_histograms,
    plot_lr_grid,
    pool_competing_counts,
)


logger = logging.getLogger(__name__)

_CSV_SPECS = (
    ("S1", "train", "train_set1.csv"),
    ("S1", "test", "test_set1.csv"),
    ("S2", "train", "train_set2.csv"),
    ("S2", "test", "test_set2.csv"),
)
_PATTERN_NAME_MAP = {
    "Word": "Word",
    "Sentence": "Sentence",
    "Complex_Event": "Complex Event",
    "Basic_Event": "Basic Event",
}
EXPECTED_EXPERIMENT_IDS = (
    "exp_1_entity_vectors",
    "exp_1_entity_vectors_attn_followup",
    "exp_1_entity_vectors_GRU_followup",
)


def load_competing_counts(data_dir: str | Path) -> dict[str, dict[str, list[int]]]:
    """Load competing-event counts from the four CSVs."""

    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"Data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Data path is not a directory: {root}")

    missing_paths = [str(root / filename) for _, _, filename in _CSV_SPECS if not (root / filename).exists()]
    if missing_paths:
        raise ValueError("Missing required competing-event CSVs:\n" + "\n".join(missing_paths))

    results: dict[str, dict[str, list[int]]] = {}
    for split, phase, filename in _CSV_SPECS:
        label = f"{split}_{phase}"
        category_counts: dict[str, list[int]] = defaultdict(list)
        csv_path = root / filename
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"consistent", "systematicity_pattern", "competing_events"}
            field_names = set(reader.fieldnames or [])
            missing_fields = sorted(required_fields - field_names)
            if missing_fields:
                raise ValueError(
                    f"{csv_path} is missing required columns: {', '.join(missing_fields)}"
                )

            for row in reader:
                if row["consistent"] != "True":
                    continue

                pattern = row["systematicity_pattern"].strip()
                if pattern not in _PATTERN_NAME_MAP:
                    continue

                competing = row["competing_events"].strip()
                if not competing:
                    continue

                category_counts[_PATTERN_NAME_MAP[pattern]].append(len(competing.split("|")))

        results[label] = dict(category_counts)

    return results


def write_competing_events_table(
    counts: dict[str, dict[str, list[int]]],
    output_dir: str | Path,
) -> None:
    """Write the pooled competing-events table used by the paper."""

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r" & \textbf{Category} & \textbf{$N$} & \textbf{Mean} & \textbf{SD} & \textbf{Median} & \textbf{Min} & \textbf{Max} \\",
        r"\midrule",
    ]

    for split in ("S1", "S2"):
        pooled = pool_competing_counts(counts, split)
        lines.append(r"\multicolumn{8}{l}{\textit{Split " + split[-1] + r"}} \\")
        lines.append(r"\addlinespace[2pt]")

        pooled_values: list[int] = []
        for category in COMPETING_EVENT_ORDER:
            if category not in pooled:
                continue
            values = np.array(pooled[category], dtype=float)
            pooled_values.extend(int(value) for value in values)
            sd = values.std(ddof=1) if len(values) > 1 else float("nan")
            lines.append(
                f" & {category} & {len(values)}"
                f" & {values.mean():.1f} & {sd:.1f}"
                f" & {np.median(values):.0f}"
                f" & {values.min():.0f} & {values.max():.0f} \\\\"
            )

        all_values = np.array(pooled_values, dtype=float)
        all_sd = all_values.std(ddof=1) if len(all_values) > 1 else float("nan")
        lines.append(r"\cmidrule(l){2-8}")
        lines.append(
            f" & All & {len(all_values)}"
            f" & {all_values.mean():.1f} & {all_sd:.1f}"
            f" & {np.median(all_values):.0f}"
            f" & {all_values.min():.0f} & {all_values.max():.0f} \\\\"
        )
        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Number of competing events per sentence by test category and data split, pooled across train and test within each split. Word/Sentence/Complex/Basic: systematicity test groups with increasing compositional difficulty. Competitors are produced by substituting one semantic slot at a time in the described event(s).}",
            r"\label{tab:competing_events_combined}",
            r"\end{table}",
        ]
    )

    _write_text(Path(output_dir) / "stat_competing_events_combined.tex", lines)


def write_competing_events_table_by_phase(
    counts: dict[str, dict[str, list[int]]],
    output_dir: str | Path,
) -> None:
    """Write the by-phase competing-events table used by the paper."""

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}lllrrrrrr@{}}",
        r"\toprule",
        r" & \textbf{Category} & \textbf{Phase} & \textbf{$N$} & \textbf{Mean} & \textbf{SD} & \textbf{Median} & \textbf{Min} & \textbf{Max} \\",
        r"\midrule",
    ]

    for split in ("S1", "S2"):
        lines.append(r"\multicolumn{9}{l}{\textit{Split " + split[-1] + r"}} \\")
        lines.append(r"\addlinespace[2pt]")

        for category in COMPETING_EVENT_ORDER:
            first_phase = True
            wrote_category = False
            for phase in ("train", "test"):
                label = f"{split}_{phase}"
                if label not in counts or category not in counts[label]:
                    continue

                values = np.array(counts[label][category], dtype=float)
                if len(values) == 0:
                    continue

                sd = values.std(ddof=1) if len(values) > 1 else float("nan")
                category_label = category if first_phase else ""
                lines.append(
                    f" & {category_label} & {phase.capitalize()}"
                    f" & {len(values)} & {values.mean():.1f} & {sd:.1f}"
                    f" & {np.median(values):.0f}"
                    f" & {values.min():.0f} & {values.max():.0f} \\\\"
                )
                first_phase = False
                wrote_category = True

            if wrote_category:
                lines.append(r"\addlinespace[2pt]")

        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Number of competing events per sentence disaggregated by train/test phase and data split. This shows how the competitor distribution differs between sentences used for training and those reserved for systematicity testing.}",
            r"\label{tab:competing_events_combined_by_phase}",
            r"\end{table}",
        ]
    )

    _write_text(Path(output_dir) / "stat_competing_events_combined_by_phase.tex", lines)


def run_competing_events_analysis(
    data_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Write the competing-events appendix tables and plots."""

    counts = load_competing_counts(data_dir)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    plot_competitor_histograms(counts, path / "competing_events_combined_hist.png")
    plot_competitor_histograms(
        counts,
        path / "competing_events_combined_hist_by_split.png",
        pool_train_test=False,
    )
    plot_competitor_boxplots(counts, path / "competing_events_combined_box.png")
    plot_competitor_boxplots(
        counts,
        path / "competing_events_combined_box_by_split.png",
        pool_train_test=False,
    )
    write_competing_events_table(counts, path)
    write_competing_events_table_by_phase(counts, path)


def _write_text(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {output_path}")


def discover_trajectory_jsons(trajectory_root: str | Path) -> list[Path]:
    """Find the full set of training-trajectory JSON files used by the appendix plots."""

    root = Path(trajectory_root)
    if not root.exists():
        raise ValueError(f"Training-trajectory root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Training-trajectory root is not a directory: {root}")

    missing_dirs = [
        str(root / experiment_id)
        for experiment_id in EXPECTED_EXPERIMENT_IDS
        if not (root / experiment_id).is_dir()
    ]
    if missing_dirs:
        raise ValueError(
            "Missing required training-trajectory directories:\n" + "\n".join(missing_dirs)
        )

    json_paths: list[Path] = []
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        experiment_dir = root / experiment_id
        experiment_paths = sorted(experiment_dir.glob("*.json"))
        if not experiment_paths:
            raise ValueError(f"No trajectory JSONs found in {experiment_dir}")
        json_paths.extend(experiment_paths)

    return json_paths


def build_trajectory_dict(
    trajectory_paths: list[Path],
) -> dict[str, list[dict[str, object]]]:
    """Build architecture-grouped LR/validation trajectories from JSON files."""

    payloads = [
        (trajectory_path, json.loads(trajectory_path.read_text()))
        for trajectory_path in trajectory_paths
    ]
    default_hidden_dims = _detect_default_hidden_dims_from_payloads(payloads)
    trajectories: dict[str, list[dict[str, object]]] = defaultdict(list)

    for trajectory_path, payload in payloads:
        model_name = str(payload["model_name"])
        parsed = parse_checkpoint_name(model_name)
        arch_family = str(parsed["arch_family"])
        hidden_dim = int(parsed["hidden_dim"])
        arch_name = make_architecture_label(
            arch_family,
            hidden_dim,
            default_hidden_dims[arch_family],
        )

        history = payload["training_history"]
        if not isinstance(history, dict):
            raise ValueError(f"training_history is not a dict in: {trajectory_path}")
        if "learning_rate" not in history:
            raise ValueError(f"Training history is missing learning_rate: {trajectory_path}")
        if "val_score" not in history:
            raise ValueError(f"Training history is missing val_score: {trajectory_path}")
        if not history["learning_rate"]:
            raise ValueError(f"Training history has empty learning_rate: {trajectory_path}")
        if not history["val_score"]:
            raise ValueError(f"Training history has empty val_score: {trajectory_path}")

        trajectories[arch_name].append(
            {
                "lr": [float(value) for value in history["learning_rate"]],
                "val_score": [float(value) for value in history["val_score"]],
                "name": model_name,
            }
        )

    return dict(trajectories)


def run_lr_schedule_analysis(
    trajectory_root: str | Path,
    output_dir: str | Path,
) -> None:
    """Write the LR-schedule and best-epoch appendix figures."""

    trajectory_paths = discover_trajectory_jsons(trajectory_root)
    trajectories = build_trajectory_dict(trajectory_paths)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_lr_grid(trajectories, output_path / "lr_schedules.png")
    plot_best_epoch_boxplot(trajectories, output_path / "best_epoch_boxplot.png")


def _detect_default_hidden_dims_from_payloads(
    payloads: list[tuple[Path, dict[str, object]]],
) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for _, payload in payloads:
        model_name = str(payload["model_name"])
        parsed = parse_checkpoint_name(model_name)
        counts[str(parsed["arch_family"])][int(parsed["hidden_dim"])] += 1
    return {family: counter.most_common(1)[0][0] for family, counter in counts.items()}
