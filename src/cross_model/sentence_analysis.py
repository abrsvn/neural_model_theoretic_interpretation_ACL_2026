"""Sentence-level distribution and descriptive appendix outputs."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import (
    GROUP_DISPLAY,
    GROUP_KEYS,
    SPLITS,
    arch_display,
    is_variant_architecture,
)
from .plotting import (
    plot_advantage_boxplots,
    plot_advantage_histograms,
    plot_gap_by_split,
    pool_arch_entity,
)


logger = logging.getLogger(__name__)

ENTITIES = ("noent", "ent")


def group_advantages(
    sentence_data: pd.DataFrame,
    *,
    train_or_test: str,
) -> dict[str, dict[tuple[str, str], list[np.ndarray]]]:
    """Group per-sentence advantages into one array per model."""

    filtered = sentence_data[sentence_data["train_or_test"] == train_or_test]
    grouped: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {}
    for (group, arch, entity, _model_id), rows in filtered.groupby(
        ["group", "arch", "entity", "model_id"]
    ):
        if group not in grouped:
            grouped[group] = defaultdict(list)
        grouped[group][(arch, entity)].append(rows["advantage"].to_numpy())
    return {group: dict(values) for group, values in grouped.items()}


def write_distribution_table(
    grouped_advantages: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    archs: list[str],
    output_path: str | Path,
    *,
    label: str,
) -> None:
    """Write the advantage-distribution summary table."""

    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}lllrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Architecture} & \textbf{Ent} & \textbf{$N$} & \textbf{Mean} & \textbf{Median} & \textbf{SD} & \textbf{\%${>}$0} & \textbf{\%${>}$0.5} \\",
        r"\midrule",
    ]

    for group_index, group in enumerate(GROUP_KEYS):
        if group not in grouped_advantages:
            continue
        present_archs = _archs_in_group(grouped_advantages, group, archs)
        for arch_index, arch in enumerate(present_archs):
            for entity in ENTITIES:
                pooled = pool_arch_entity(grouped_advantages, group, arch, entity)
                if pooled is None:
                    continue
                group_label = GROUP_DISPLAY[group] if arch_index == 0 and entity == "noent" else ""
                arch_label = arch_display(arch) if entity == "noent" else ""
                entity_label = "$+$" if entity == "ent" else "$-$"
                pct_positive = 100 * (pooled > 0).mean()
                pct_above_half = 100 * (pooled > 0.5).mean()
                lines.append(
                    f"{group_label} & {arch_label} & {entity_label}"
                    f" & {len(pooled)}"
                    f" & {pooled.mean():.3f} & {np.median(pooled):.3f}"
                    f" & {pooled.std(ddof=1):.3f}"
                    f" & {pct_positive:.1f} & {pct_above_half:.1f} \\\\"
                )
            if arch_index < len(present_archs) - 1:
                lines.append(r"\addlinespace[1pt]")
        if group_index < len(GROUP_KEYS) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Per-sentence advantage score distributions by group, architecture, and entity vector condition. $N$: total sentence evaluations (sentences $\times$ models). \%${>}$0: proportion with positive advantage. \%${>}$0.5: proportion with advantage above 0.5.}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )

    _write_text(output_path, lines)


def run_distribution_analysis(
    sentence_data: pd.DataFrame,
    archs: list[str],
    output_dir: str | Path,
) -> None:
    """Write the cross-model distribution appendix outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_data = group_advantages(sentence_data, train_or_test="test")
    train_data = group_advantages(sentence_data, train_or_test="train")

    plot_advantage_histograms(test_data, archs, output_path / "advantage_histograms_test.png")
    plot_advantage_boxplots(test_data, archs, output_path / "advantage_boxplots_test.png")
    write_distribution_table(
        test_data,
        archs,
        output_path / "stat_advantage_distributions_test.tex",
        label="tab:advantage_distributions_test",
    )

    plot_advantage_histograms(train_data, archs, output_path / "advantage_histograms_train.png")
    plot_advantage_boxplots(train_data, archs, output_path / "advantage_boxplots_train.png")
    write_distribution_table(
        train_data,
        archs,
        output_path / "stat_advantage_distributions_train.tex",
        label="tab:advantage_distributions_train",
    )

    for split in SPLITS:
        split_rows = sentence_data[sentence_data["split"] == split]
        for phase in ("test", "train"):
            split_data = group_advantages(split_rows, train_or_test=phase)
            plot_advantage_boxplots(
                split_data,
                archs,
                output_path / f"advantage_boxplots_{split}_{phase}.png",
            )

    plot_gap_by_split(
        sentence_data,
        archs,
        output_path / "generalization_gap_combined_by_split.png",
    )


def _archs_in_group(
    grouped_advantages: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    group: str,
    archs: list[str],
) -> list[str]:
    if group not in grouped_advantages:
        return []
    return [
        arch
        for arch in archs
        if any((arch, entity) in grouped_advantages[group] for entity in ENTITIES)
    ]


def _write_text(output_path: str | Path, lines: list[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {path}")


def compute_obs_counts(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, str], tuple[int, int, int]]:
    """Count observations per `(group, split, train_or_test)` cell."""

    sentences: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    models: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for row in rows:
        key = (
            str(row["group"]),
            str(row["split"]),
            str(row["train_or_test"]),
        )
        sentences[key].add(str(row["sentence"]))
        models[key].add(str(row["model_id"]))
        counts[key] += 1

    return {
        key: (counts[key], len(sentences[key]), len(models[key]))
        for key in counts
    }


def compute_per_split_stats(
    rows: list[dict[str, object]],
    train_or_test: str,
) -> dict[tuple[str, str, str, str], dict[str, float | int | list[float]]]:
    """Compute per-split descriptive statistics for one train/test phase."""

    by_model: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row["train_or_test"]) != train_or_test:
            continue
        key = (
            str(row["group"]),
            str(row["split"]),
            str(row["arch"]),
            str(row["entity"]),
            str(row["model_id"]),
        )
        by_model[key].append(float(row["advantage"]))

    cell_data: dict[
        tuple[str, str, str, str],
        dict[str, list[float]],
    ] = defaultdict(lambda: {"all_obs": [], "model_means": []})

    for (group, split, arch, entity, _model_id), advantages in by_model.items():
        cell_key = (group, split, arch, entity)
        cell_data[cell_key]["all_obs"].extend(advantages)
        cell_data[cell_key]["model_means"].append(_mean(advantages))

    result: dict[tuple[str, str, str, str], dict[str, float | int | list[float]]] = {}
    for cell_key, cell in cell_data.items():
        observations = cell["all_obs"]
        model_means = cell["model_means"]
        result[cell_key] = {
            "sentence_mean": _mean(observations),
            "model_means": model_means,
            "model_mean": _mean(model_means),
            "model_se": _se(model_means),
            "n_obs": len(observations),
            "n_models": len(model_means),
        }

    return result


def compute_pooled_stats(
    rows: list[dict[str, object]],
    train_or_test: str,
) -> dict[tuple[str, str, str], dict[str, float | int]]:
    """Compute pooled descriptive statistics across S1 and S2."""

    sentence_data: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_model: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        if str(row["train_or_test"]) != train_or_test:
            continue
        pool_key = (
            str(row["group"]),
            str(row["arch"]),
            str(row["entity"]),
        )
        sentence_data[pool_key].append(float(row["advantage"]))
        by_model[pool_key + (str(row["model_id"]),)].append(float(row["advantage"]))

    model_data: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (group, arch, entity, _model_id), advantages in by_model.items():
        model_data[(group, arch, entity)].append(_mean(advantages))

    result: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for pool_key, observations in sentence_data.items():
        model_means = model_data[pool_key]
        sentence_mean = _mean(observations)
        model_mean = _mean(model_means)
        result[pool_key] = {
            "sent_mean": sentence_mean,
            "model_mean": model_mean,
            "model_se": _se(model_means),
            "n_obs": len(observations),
            "n_models": len(model_means),
            "discrepancy": sentence_mean - model_mean,
        }

    return result


def write_obs_counts_table(
    obs_counts: dict[tuple[str, str, str], tuple[int, int, int]],
    output_path: str | Path,
) -> None:
    """Write the observation-structure LaTeX table."""

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"\textbf{Group} & \textbf{Split} & \textbf{Train sent.} & \textbf{Test sent.} & \textbf{Models} \\",
        r"\midrule",
    ]

    for group_index, group in enumerate(GROUP_KEYS):
        for split_index, split in enumerate(SPLITS):
            group_label = GROUP_DISPLAY[group] if split_index == 0 else ""
            train_key = (group, split, "train")
            test_key = (group, split, "test")
            n_train_sent = obs_counts.get(train_key, (0, 0, 0))[1]
            n_test_sent = obs_counts.get(test_key, (0, 0, 0))[1]
            n_models = obs_counts.get(test_key, (0, 0, 0))[2]
            lines.append(
                f"{group_label} & {split} & {n_train_sent} & {n_test_sent} & {n_models} \\\\"
            )
        if group_index < len(GROUP_KEYS) - 1:
            lines.append(r"\addlinespace")

    total_obs = sum(value[0] for value in obs_counts.values())
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Observation structure per group and split. Total observations per cell = sentences $\times$ models."
            + f" Grand total: {total_obs:,} observations."
            + r" The asymmetry in Complex and Basic Event sentence counts arises from the complementary exclusion structure.}",
            r"\label{tab:obs_counts}",
            r"\end{table}",
        ]
    )

    _write_text(output_path, lines)


def write_per_split_table(
    per_split: dict[tuple[str, str, str, str], dict[str, float | int | list[float]]],
    archs: list[str],
    output_path: str | Path,
    *,
    train_or_test: str,
) -> None:
    """Write the longtable of per-split descriptive statistics."""

    n_seeds = max((int(value["n_models"]) for value in per_split.values()), default=0)
    lines = [
        r"\scriptsize",
        r"\begin{longtable}{@{}llrrrrrrrr@{}}",
        r"\caption{Per-split descriptive statistics"
        + f" ({train_or_test} sentences)."
        + r" Mean advantage by group, architecture, split, and entity vector condition."
        + f" SE$_{{+}}$: standard error for the $+$ent condition across {n_seeds} random seeds."
        + r" $\Delta$: entity vector effect ($+$ent $-$ $-$ent).}",
        f"\\label{{tab:descriptive_per_split_{train_or_test}}} \\\\",
        r"\toprule",
        r" &  & \multicolumn{4}{c}{\textbf{Split 1}} & \multicolumn{4}{c}{\textbf{Split 2}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Group} & \textbf{Architecture} & $-$ent & $+$ent & SE$_{+}$ & $\Delta$ & $-$ent & $+$ent & SE$_{+}$ & $\Delta$ \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{10}{c}{\scriptsize\textit{(continued)}} \\",
        r"\toprule",
        r" &  & \multicolumn{4}{c}{\textbf{Split 1}} & \multicolumn{4}{c}{\textbf{Split 2}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Group} & \textbf{Architecture} & $-$ent & $+$ent & SE$_{+}$ & $\Delta$ & $-$ent & $+$ent & SE$_{+}$ & $\Delta$ \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule \multicolumn{10}{r@{}}{\scriptsize\textit{(continued on next page)}} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for group_index, group in enumerate(GROUP_KEYS):
        for arch_index, arch in enumerate(archs):
            group_label = GROUP_DISPLAY[group] if arch_index == 0 else ""
            row_parts = [group_label, arch_display(arch)]

            for split in SPLITS:
                no_entity_stats = per_split.get((group, split, arch, "noent"))
                with_entity_stats = per_split.get((group, split, arch, "ent"))
                if no_entity_stats and with_entity_stats:
                    no_entity_mean = float(no_entity_stats["sentence_mean"])
                    with_entity_mean = float(with_entity_stats["sentence_mean"])
                    with_entity_se = float(with_entity_stats["model_se"])
                    row_parts.extend(
                        [
                            _fmt(no_entity_mean),
                            _fmt(with_entity_mean),
                            _fmt(with_entity_se),
                            _fmt_delta(with_entity_mean - no_entity_mean),
                        ]
                    )
                else:
                    row_parts.extend(["--", "--", "--", "--"])

            lines.append(" & ".join(row_parts) + r" \\")

        if group_index < len(GROUP_KEYS) - 1:
            lines.append(r"\addlinespace")

    lines.extend([r"\end{longtable}", r"\normalsize"])
    _write_text(output_path, lines)


def write_pooled_table(
    pooled: dict[tuple[str, str, str], dict[str, float | int]],
    archs: list[str],
    output_path: str | Path,
    *,
    train_or_test: str,
) -> None:
    """Write the pooled sentence-level vs model-level descriptive table."""

    n_splits, n_seeds = _pooled_counts(pooled)
    n_models = n_splits * n_seeds

    def pooled_arch_label(arch: str) -> str:
        return arch_display(arch).replace("Attn ", "")

    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r" &  & \multicolumn{3}{c}{\textbf{Sentence-level}} & \multicolumn{4}{c}{\textbf{Model-level}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-9}",
        r"\textbf{Group} & \textbf{Architecture} & $-$ent & $+$ent & $\Delta$ & $-$ent & $+$ent & SE$_{+}$ & $\Delta$ \\",
        r"\midrule",
    ]

    for group_index, group in enumerate(GROUP_KEYS):
        for arch in archs:
            group_label = GROUP_DISPLAY[group] if arch == archs[0] else ""
            no_entity_stats = pooled.get((group, arch, "noent"))
            with_entity_stats = pooled.get((group, arch, "ent"))
            if no_entity_stats and with_entity_stats:
                sentence_no_entity = float(no_entity_stats["sent_mean"])
                sentence_with_entity = float(with_entity_stats["sent_mean"])
                model_no_entity = float(no_entity_stats["model_mean"])
                model_with_entity = float(with_entity_stats["model_mean"])
                row = (
                    f"{group_label} & {pooled_arch_label(arch)}"
                    f" & {_fmt(sentence_no_entity)} & {_fmt(sentence_with_entity)}"
                    f" & {_fmt_delta(sentence_with_entity - sentence_no_entity)}"
                    f" & {_fmt(model_no_entity)} & {_fmt(model_with_entity)}"
                    f" & {_fmt(float(with_entity_stats['model_se']))}"
                    f" & {_fmt_delta(model_with_entity - model_no_entity)} \\\\"
                )
            else:
                row = (
                    f"{group_label} & {pooled_arch_label(arch)}"
                    r" & -- & -- & -- & -- & -- & -- & -- \\"
                )
            lines.append(row)

        if group_index < len(GROUP_KEYS) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Pooled descriptive statistics"
            + f" ({train_or_test} sentences, both splits combined)."
            + r" Sentence-level: mean weighted by observations"
            + f" (splits with more {train_or_test} sentences count more)."
            + r" Model-level: mean weighted equally per model"
            + f" (each of {n_models} models counts once)."
            + f" SE$_{{+}}$: standard error for the $+$ent condition across {n_models} models ({n_seeds} seeds $\\times$ {n_splits} splits)."
            + r" $\Delta$: $+$ent $-$ $-$ent.}",
            f"\\label{{tab:descriptive_pooled_{train_or_test}}}",
            r"\end{table}",
        ]
    )
    _write_text(output_path, lines)


def write_paper_table(
    pooled: dict[tuple[str, str, str], dict[str, float | int]],
    main_archs: list[str],
    output_path: str | Path,
    *,
    weighting: str,
) -> None:
    """Write the compact paper-format main-results table."""

    mean_key = "sent_mean" if weighting == "sentence" else "model_mean"
    n_splits, n_seeds = _pooled_counts(pooled)
    n_models = n_splits * n_seeds

    if weighting == "sentence":
        caption = (
            r"\caption{\small Experiment 1: test-only advantage (higher is better) by "
            r"architecture, test group and $\pm$entity, aggregated across all test "
            r"sentences including modifier variants. GRU \& LSTM $+$ent achieve the "
            r"highest Basic Event scores (\textbf{0.78} and 0.77; statistically "
            r"indistinguishable: $p = 1.00$). See Table~\ref{tab:complexity_breakdown} "
            r"for how modifier complexity affects these aggregate scores.}"
        )
        label = "tab:main_results"
    else:
        caption = (
            r"\caption{Test-only advantage scores (described $-$ competing) by architecture, "
            r"test group, and $\pm$entity."
            + (
            f" Model-level means averaged over {n_models} models"
            + f" ({n_splits} splits $\\times$ {n_seeds} seeds)."
            )
            + r" Higher is better.}"
        )
        label = "tab:main_results_model"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\vspace{-0.5em}" if weighting == "sentence" else None,
        r"\small",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Arch} & \textbf{Ent} & \textbf{Word} & \textbf{Sent} & \textbf{Cmplx} & \textbf{Basic} \\",
        r"\midrule",
    ]

    for arch_index, arch in enumerate(main_archs):
        for entity_index, entity in enumerate(ENTITIES):
            entity_label = "$-$ent" if entity == "noent" else "$+$ent"
            arch_label = r"\multirow{2}{*}{" + arch_display(arch) + "}" if entity_index == 0 else ""
            cells = [arch_label, entity_label]
            for group in GROUP_KEYS:
                stats = pooled.get((group, arch, entity))
                cells.append(f"{float(stats[mean_key]):.2f}" if stats else "--")
            lines.append(" & ".join(cells) + r" \\")
        if arch_index < len(main_archs) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{-0.7em}" if weighting == "sentence" else None,
            caption,
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    _write_text(output_path, [line for line in lines if line is not None])


def run_descriptive_analysis(
    rows: list[dict[str, object]],
    archs: list[str],
    output_dir: str | Path,
    paper_sentence_output_path: str | Path,
) -> None:
    """Write all descriptive appendix tables to `output_dir`."""

    output_path = Path(output_dir)
    paper_output_path = Path(paper_sentence_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    paper_output_path.parent.mkdir(parents=True, exist_ok=True)
    main_archs = [arch for arch in archs if not is_variant_architecture(arch)]

    obs_counts = compute_obs_counts(rows)
    per_split_test = compute_per_split_stats(rows, "test")
    per_split_train = compute_per_split_stats(rows, "train")
    pooled_test = compute_pooled_stats(rows, "test")
    pooled_train = compute_pooled_stats(rows, "train")

    write_obs_counts_table(obs_counts, output_path / "stat_obs_counts.tex")
    write_per_split_table(
        per_split_test,
        archs,
        output_path / "stat_descriptive_per_split_test.tex",
        train_or_test="test",
    )
    write_per_split_table(
        per_split_train,
        archs,
        output_path / "stat_descriptive_per_split_train.tex",
        train_or_test="train",
    )
    write_pooled_table(
        pooled_test,
        archs,
        output_path / "stat_descriptive_pooled_test.tex",
        train_or_test="test",
    )
    write_pooled_table(
        pooled_train,
        archs,
        output_path / "stat_descriptive_pooled_train.tex",
        train_or_test="train",
    )
    write_paper_table(
        pooled_test,
        main_archs,
        output_path / "stat_descriptive_paper.tex",
        weighting="model",
    )
    write_paper_table(
        pooled_test,
        main_archs,
        paper_output_path,
        weighting="sentence",
    )


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _fmt(value: float, *, decimals: int = 3, sign: bool = False) -> str:
    if math.isnan(value):
        return "--"
    format_spec = f"+.{decimals}f" if sign else f".{decimals}f"
    return f"{value:{format_spec}}"


def _fmt_delta(value: float) -> str:
    return _fmt(value, sign=True)


def _pooled_counts(
    pooled: dict[tuple[str, str, str], dict[str, float | int]],
) -> tuple[int, int]:
    n_models = max((int(value["n_models"]) for value in pooled.values()), default=0)
    n_splits = len(SPLITS)
    n_seeds = n_models // n_splits if n_splits else n_models
    return (n_splits, n_seeds)
