"""CLI for evaluation and result regeneration."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from checkpoints import (
    DEFAULT_METADATA_DIR,
    build_model,
    load_checkpoint_state,
    load_model_weights,
    lookup_run,
    run_from_checkpoint_state,
    resolve_training_history,
)
from data.dataset import load_sentence_records
from data.targets import TargetBuilder, TargetWeights
from evaluation.reporting import build_systematicity_summary_rows
from evaluation.systematicity import (
    evaluate_systematicity_suite,
    write_systematicity_rows_csv,
    write_systematicity_summary_csv,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the CLI."""

    parser = argparse.ArgumentParser(prog="python -m cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_evaluate_parser(subparsers)
    _add_plot_entity_parser(subparsers)
    _add_plot_entity_detailed_parser(subparsers)
    _add_plot_gap_parser(subparsers)
    _add_paper_table_parser(subparsers)
    _add_competing_events_parser(subparsers)
    _add_plot_training_parser(subparsers)
    _add_lr_schedules_parser(subparsers)
    _add_sentence_data_parser(subparsers)
    _add_distribution_parser(subparsers)
    _add_descriptive_parser(subparsers)

    args = parser.parse_args()
    if args.command == "evaluate":
        _run_evaluate(args)
        return
    if args.command == "plot-entity":
        _run_plot_entity(args)
        return
    if args.command == "plot-entity-detailed":
        _run_plot_entity_detailed(args)
        return
    if args.command == "plot-gap":
        _run_plot_gap(args)
        return
    if args.command == "paper-table":
        _run_paper_table(args)
        return
    if args.command == "competing-events":
        _run_competing_events(args)
        return
    if args.command == "plot-training":
        _run_plot_training(args)
        return
    if args.command == "lr-schedules":
        _run_lr_schedules(args)
        return
    if args.command == "distribution":
        _run_distribution(args)
        return
    if args.command == "descriptive":
        _run_descriptive(args)
        return
    if args.command == "sentence-data":
        _run_sentence_data(args)
        return


def _configure_torch_threads_from_env() -> None:
    thread_text = os.environ.get("TORCH_NUM_THREADS")
    if thread_text is not None:
        torch.set_num_threads(int(thread_text))

    interop_text = os.environ.get("TORCH_NUM_INTEROP_THREADS")
    if interop_text is not None:
        torch.set_num_interop_threads(int(interop_text))


def _run_evaluate(args: argparse.Namespace) -> None:
    _configure_torch_threads_from_env()
    repo_root = _resolve_repo_root(args.repo_root)
    run_spec = lookup_run(
        experiment_id=args.experiment_id,
        model_type=args.model_type,
        entity_condition=args.entity_condition,
        split=args.split,
        model_index=args.model_index,
        metadata_dir=args.metadata_dir,
    )
    model = build_model(run_spec)
    load_model_weights(model, args.checkpoint_path, device=args.device)
    target_builder = _build_target_builder(
        repo_root=repo_root,
        with_entity_vectors=run_spec.concat_entity_vector,
    )

    train_records = load_sentence_records(repo_root / "data" / f"train_set{run_spec.split}.csv")
    test_records = load_sentence_records(repo_root / "data" / f"test_set{run_spec.split}.csv")
    results = evaluate_systematicity_suite(
        model,
        train_records=train_records,
        test_records=test_records,
        target_builder=target_builder,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    checkpoint_stem = Path(args.checkpoint_path).stem
    write_systematicity_rows_csv(
        results,
        output_dir / f"{checkpoint_stem}_rows_extended.csv",
        canonical_only=False,
    )
    write_systematicity_rows_csv(
        results,
        output_dir / f"{checkpoint_stem}_rows_canonical.csv",
        canonical_only=True,
    )
    write_systematicity_summary_csv(
        results,
        output_dir / f"{checkpoint_stem}_group_summary.csv",
    )
    summary_rows = build_systematicity_summary_rows(run_spec, results)
    _write_csv_rows(summary_rows, output_dir / f"{checkpoint_stem}_run_summary.csv")

    print(f"Wrote evaluation outputs to {output_dir.resolve()}")


def _run_plot_entity(args: argparse.Namespace) -> None:
    from plots.paper_plots import plot_entity_vector_comparison

    summary_rows = _read_csv_rows(args.summary_csvs)
    plot_entity_vector_comparison(
        summary_rows,
        args.output_path,
        experiment_id=args.experiment_id,
        show_title=not args.omit_title,
    )
    print(f"Wrote entity comparison figure to {Path(args.output_path).resolve()}")


def _run_plot_entity_detailed(args: argparse.Namespace) -> None:
    from plots.paper_detailed_plots import plot_entity_vector_comparison_detailed

    detail_rows = _read_detail_csv_rows(args.detail_csvs)
    plot_entity_vector_comparison_detailed(
        detail_rows,
        args.output_path,
        experiment_id=args.experiment_id,
    )
    print(f"Wrote detailed entity comparison figure to {Path(args.output_path).resolve()}")


def _run_plot_gap(args: argparse.Namespace) -> None:
    from plots.paper_plots import plot_generalization_gap

    summary_rows = _read_csv_rows(args.summary_csvs)
    output_paths = plot_generalization_gap(
        summary_rows,
        args.output_dir,
        experiment_id=args.experiment_id,
    )
    print(f"Wrote {len(output_paths)} generalization-gap figures to {Path(args.output_dir).resolve()}")


def _run_paper_table(args: argparse.Namespace) -> None:
    from plots.paper_table import write_main_results_table

    summary_rows = _read_csv_rows(args.summary_csvs)
    write_main_results_table(
        summary_rows,
        args.output_path,
        experiment_id=args.experiment_id,
    )
    print(f"Wrote paper table to {Path(args.output_path).resolve()}")


def _run_competing_events(args: argparse.Namespace) -> None:
    from cross_model.diagnostics import run_competing_events_analysis

    run_competing_events_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote competing-events appendix outputs to {Path(args.output_dir).resolve()}")


def _run_sentence_data(args: argparse.Namespace) -> None:
    from cross_model.sentence_data import (
        build_sentence_dataset,
        detect_hidden_dim_defaults,
        summarize_dataset,
        write_sentence_data_csv,
    )

    analysis_dirs = [Path(path_text) for path_text in args.analysis_dirs]
    default_hidden_dims = detect_hidden_dim_defaults(analysis_dirs)
    rows = build_sentence_dataset(
        analysis_dirs,
        default_hidden_dims=default_hidden_dims,
    )
    summary = summarize_dataset(rows)
    arch_rank = {arch: index for index, arch in enumerate(summary["archs"])}
    rows.sort(key=lambda row: arch_rank[str(row["arch"])])
    write_sentence_data_csv(rows, args.output_path)
    print(
        "Wrote sentence_data.csv to "
        f"{Path(args.output_path).resolve()} "
        f"({summary['n_rows']} rows, {len(summary['archs'])} architectures, "
        f"{len(summary['models'])} models)"
    )


def _add_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("evaluate")
    _add_run_spec_arguments(parser)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help=(
            "Path to the checkout containing data/ and weights/. "
            "The default assumes an editable install backed by this checkout."
        ),
    )
    parser.add_argument(
        "--metadata-dir",
        default=str(DEFAULT_METADATA_DIR),
        help=(
            "Metadata directory containing experiments.json and checkpoint_seeds.json. "
            "The default assumes an editable install backed by this checkout."
        ),
    )
    parser.add_argument("--device", default="cpu")


def _add_plot_entity_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plot-entity")
    parser.add_argument("--summary-csvs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--omit-title",
        action="store_true",
        help="Write the figure without the top title so the output is paper-ready as-is.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional only when the supplied summary CSVs do not mix overlapping experiments.",
    )


def _add_plot_entity_detailed_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("plot-entity-detailed")
    parser.add_argument("--detail-csvs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional only when the supplied detail CSVs do not mix overlapping experiments.",
    )


def _add_plot_gap_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plot-gap")
    parser.add_argument("--summary-csvs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional only when the supplied summary CSVs do not mix overlapping experiments.",
    )


def _add_paper_table_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("paper-table")
    parser.add_argument("--summary-csvs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional only when the supplied summary CSVs do not mix overlapping experiments.",
    )


def _add_competing_events_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("competing-events")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)


def _add_sentence_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("sentence-data")
    parser.add_argument("--analysis-dirs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)


def _add_run_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--entity-condition", required=True)
    parser.add_argument("--split", required=True, type=int)
    parser.add_argument("--model-index", required=True, type=int)


def _build_target_builder(
    *,
    repo_root: str | Path,
    with_entity_vectors: bool,
) -> TargetBuilder:
    root = Path(repo_root)
    proposition_path = root / "weights" / "competitive_150_props.npz"
    entity_path = root / "weights" / "competitive_150_entities.npz"
    missing_paths = [
        str(path)
        for path in (proposition_path, entity_path)
        if not path.is_file()
    ]
    if missing_paths:
        raise ValueError(
            "The path passed via --repo-root is missing required weight files:\n"
            + "\n".join(missing_paths)
            + "\nRun from the checkout or pass --repo-root explicitly."
        )

    target_weights = TargetWeights.from_paths(
        proposition_path,
        entity_path,
    )
    return TargetBuilder(target_weights, with_entity_vectors=with_entity_vectors)


def _resolve_repo_root(repo_root: str | Path) -> Path:
    root = Path(repo_root)
    missing_dirs = [
        str(root / name)
        for name in ("data", "weights")
        if not (root / name).is_dir()
    ]
    if missing_dirs:
        raise ValueError(
            "The path passed via --repo-root is missing required directories:\n"
            + "\n".join(missing_dirs)
            + "\nRun from the checkout or pass --repo-root explicitly."
        )
    return root


def _read_csv_rows(csv_paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    return rows


def _read_detail_csv_rows(csv_paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path_text in csv_paths:
        csv_path = Path(csv_path_text)
        if not csv_path.name.endswith("_rows_extended.csv"):
            raise ValueError(f"Detailed plotting requires *_rows_extended.csv inputs: {csv_path}")

        experiment_id = csv_path.parent.parent.name
        run_dir_parts = csv_path.parent.name.split("__")
        if len(run_dir_parts) != 4:
            raise ValueError(
                "Detailed plotting requires run directories named "
                f"<MODEL_TYPE>__<ENTITY_CONDITION>__s<SPLIT>__m<MODEL_INDEX>: {csv_path.parent.name}"
            )

        model_type, entity_condition, split_token, model_index_token = run_dir_parts
        if not split_token.startswith("s") or not model_index_token.startswith("m"):
            raise ValueError(
                "Detailed plotting requires run directories named "
                f"<MODEL_TYPE>__<ENTITY_CONDITION>__s<SPLIT>__m<MODEL_INDEX>: {csv_path.parent.name}"
            )

        run_split = int(split_token[1:])
        model_index = int(model_index_token[1:])

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                detail_row: dict[str, object] = dict(row)
                detail_row["experiment_id"] = experiment_id
                detail_row["model_type"] = model_type
                detail_row["entity_condition"] = entity_condition
                detail_row["run_split"] = run_split
                detail_row["model_index"] = model_index
                rows.append(detail_row)
    return rows


def _write_csv_rows(rows: list[dict[str, object]], output_path: str | Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})



def _run_plot_training(args: argparse.Namespace) -> None:
    from plots.paper_plots import plot_training_curves
    from evaluation.reporting import build_training_history_record

    history_records = []
    trajectory_root = Path(args.trajectory_root) if args.trajectory_root else Path("training_trajectories")
    for checkpoint_path_text in args.checkpoint_paths:
        checkpoint_path = Path(checkpoint_path_text)
        checkpoint_state = load_checkpoint_state(checkpoint_path)
        run_spec = run_from_checkpoint_state(checkpoint_state)
        history = resolve_training_history(
            checkpoint_state,
            trajectory_root=trajectory_root,
        )
        history_records.append(build_training_history_record(run_spec, history))

    output_paths = plot_training_curves(
        history_records,
        args.output_dir,
        experiment_id=args.experiment_id,
    )
    print(f"Wrote {len(output_paths)} training-curve figures to {Path(args.output_dir).resolve()}")

def _run_lr_schedules(args: argparse.Namespace) -> None:
    from cross_model.diagnostics import run_lr_schedule_analysis

    run_lr_schedule_analysis(
        trajectory_root=args.trajectory_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote LR-schedule appendix outputs to {Path(args.output_dir).resolve()}")

def _add_plot_training_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("plot-training")
    parser.add_argument("--checkpoint-paths", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--trajectory-root", default=None)

def _add_lr_schedules_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("lr-schedules")
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--output-dir", required=True)

def _run_distribution(args: argparse.Namespace) -> None:
    import pandas as pd
    from cross_model.metadata import sort_architectures
    from cross_model.sentence_analysis import run_distribution_analysis

    df = pd.read_csv(args.sentence_csv)
    archs = sort_architectures(df["arch"].unique().tolist())
    run_distribution_analysis(df, archs, args.output_dir)
    print(f"Wrote distribution plots and tables to {Path(args.output_dir).resolve()}")

def _run_descriptive(args: argparse.Namespace) -> None:
    from cross_model.metadata import sort_architectures
    from cross_model.sentence_analysis import run_descriptive_analysis

    rows = _read_csv_rows([args.sentence_csv])
    archs = sort_architectures(list(dict.fromkeys(row["arch"] for row in rows)))
    run_descriptive_analysis(
        rows,
        archs,
        args.output_dir,
        args.paper_sentence_output_path,
    )
    print(f"Wrote descriptive tables to {Path(args.output_dir).resolve()}")

def _add_distribution_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("distribution")
    parser.add_argument("--sentence-csv", required=True)
    parser.add_argument("--output-dir", required=True)

def _add_descriptive_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("descriptive")
    parser.add_argument("--sentence-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-sentence-output-path", required=True)

if __name__ == "__main__":
    main()
