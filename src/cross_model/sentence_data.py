"""Cross-model sentence-data aggregation for appendix statistics."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .metadata import (
    GROUP_DISPLAY,
    GROUP_KEYS,
    entity_label,
    make_architecture_label,
    parse_checkpoint_name,
    sort_architectures,
)


OUTPUT_COLUMNS = [
    "sentence",
    "model_id",
    "arch",
    "entity",
    "split",
    "seed",
    "hidden_dim",
    "group",
    "train_or_test",
    "described_score",
    "competing_score",
    "advantage",
]


def extract_model_metadata(model_name: str) -> dict[str, int | str]:
    """Parse one model stem into cross-model aggregation metadata."""

    parsed = parse_checkpoint_name(model_name)
    return {
        "arch_family": parsed["arch_family"],
        "hidden_dim": parsed["hidden_dim"],
        "entity": entity_label(str(parsed["entity_code"])),
        "split": f"S{parsed['split']}",
        "seed": parsed["model_index"],
    }


def discover_model_csvs(analysis_dir: Path) -> list[tuple[Path, str]]:
    """Find one extended row CSV per reevaluated run directory."""

    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory does not exist: {analysis_dir}")
    if not analysis_dir.is_dir():
        raise ValueError(f"Analysis path is not a directory: {analysis_dir}")

    run_dirs = sorted(path for path in analysis_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found in {analysis_dir}")

    results: list[tuple[Path, str]] = []
    problems: list[str] = []
    suffix = "_best_model_rows_extended.csv"

    for run_dir in run_dirs:
        matches = sorted(run_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            problems.append(
                f"{run_dir}: expected exactly 1 *{suffix} file, found {len(matches)}"
            )
            continue
        csv_path = matches[0]
        model_name = csv_path.name.removesuffix(suffix)
        results.append((csv_path, model_name))

    if problems:
        raise ValueError("\n".join(problems))

    return results


def extract_rows_from_csv(
    csv_path: Path,
    *,
    model_id: str,
    arch: str,
    entity: str,
    split: str,
    seed: int,
    hidden_dim: int,
) -> list[dict[str, object]]:
    """Read one `*_rows_extended.csv` and return long-format sentence rows."""

    rows: list[dict[str, object]] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "group_name",
            "output_key",
            "split",
            "sentence",
            "described_score",
            "competing_score",
        }
        field_names = set(reader.fieldnames or [])
        missing_fields = sorted(required_fields - field_names)
        if missing_fields:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing_fields)}"
            )

        for row in reader:
            output_key = str(row["output_key"]).strip()
            if output_key not in GROUP_KEYS:
                raise ValueError(f"{csv_path} has unsupported output_key={output_key!r}")

            group_name = str(row["group_name"]).strip()
            expected_group = GROUP_DISPLAY[output_key]
            if group_name != expected_group.replace(" ", "_") and group_name != expected_group:
                raise ValueError(
                    f"{csv_path} has mismatched group_name={group_name!r} for output_key={output_key!r}"
                )

            train_or_test = str(row["split"]).strip()
            if train_or_test not in {"train", "test"}:
                raise ValueError(
                    f"{csv_path} has unsupported split={train_or_test!r}"
                )

            described = float(row["described_score"])
            competing = float(row["competing_score"])
            rows.append(
                {
                    "sentence": str(row["sentence"]),
                    "model_id": model_id,
                    "arch": arch,
                    "entity": entity,
                    "split": split,
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "group": output_key,
                    "train_or_test": train_or_test,
                    "described_score": f"{described:.6f}",
                    "competing_score": f"{competing:.6f}",
                    "advantage": f"{described - competing:.6f}",
                }
            )

    return rows


def build_sentence_dataset(
    analysis_dirs: list[Path],
    *,
    default_hidden_dims: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Build the long-format sentence dataset from reevaluated run CSVs."""

    all_rows: list[dict[str, object]] = []
    for analysis_dir in analysis_dirs:
        for csv_path, model_name in discover_model_csvs(analysis_dir):
            metadata = extract_model_metadata(model_name)
            arch_family = str(metadata["arch_family"])
            hidden_dim = int(metadata["hidden_dim"])
            if default_hidden_dims is not None and arch_family in default_hidden_dims:
                arch = make_architecture_label(
                    arch_family,
                    hidden_dim,
                    default_hidden_dims[arch_family],
                )
            else:
                arch = arch_family

            model_id = f"{arch}_{metadata['entity']}_{metadata['split']}_{metadata['seed']}"
            all_rows.extend(
                extract_rows_from_csv(
                    csv_path,
                    model_id=model_id,
                    arch=arch,
                    entity=str(metadata["entity"]),
                    split=str(metadata["split"]),
                    seed=int(metadata["seed"]),
                    hidden_dim=hidden_dim,
                )
            )
    return all_rows


def summarize_dataset(rows: list[dict[str, object]]) -> dict[str, object]:
    """Compute high-level summary counts for a built sentence dataset."""

    architectures: set[str] = set()
    groups: set[str] = set()
    model_ids: set[str] = set()
    train_or_tests: set[str] = set()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    sentences: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_models: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        arch = str(row["arch"])
        group = str(row["group"])
        train_or_test = str(row["train_or_test"])
        model_id = str(row["model_id"])
        sentence = str(row["sentence"])
        key = (group, train_or_test)

        architectures.add(arch)
        groups.add(group)
        model_ids.add(model_id)
        train_or_tests.add(train_or_test)
        counts[key] += 1
        sentences[key].add(sentence)
        group_models[key].add(model_id)

    per_group: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (group, train_or_test), n_rows in counts.items():
        per_group[group][train_or_test] = {
            "n_obs": n_rows,
            "n_sentences": len(sentences[(group, train_or_test)]),
            "n_models": len(group_models[(group, train_or_test)]),
        }

    return {
        "n_rows": len(rows),
        "archs": sort_architectures(list(architectures)),
        "groups": sorted(groups),
        "models": sorted(model_ids),
        "train_or_tests": sorted(train_or_tests),
        "per_group": dict(per_group),
    }


def detect_hidden_dim_defaults(analysis_dirs: list[Path]) -> dict[str, int]:
    """Detect default hidden dimensions for architecture families with variants."""

    dims_per_arch: dict[str, Counter[int]] = defaultdict(Counter)
    for analysis_dir in analysis_dirs:
        for _csv_path, model_name in discover_model_csvs(analysis_dir):
            metadata = extract_model_metadata(model_name)
            dims_per_arch[str(metadata["arch_family"])][int(metadata["hidden_dim"])] += 1

    return {
        arch_family: counts.most_common(1)[0][0]
        for arch_family, counts in dims_per_arch.items()
        if len(counts) > 1
    }


def write_sentence_data_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Write the long-format sentence dataset to a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
