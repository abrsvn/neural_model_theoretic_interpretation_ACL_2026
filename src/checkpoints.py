"""Run lookup, model construction, and checkpoint loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data.vocabulary import VOCAB
from models.attention import AttentionParams, SimpleAttentionModel
from models.recurrent import RecurrentParams, SimpleGRU, SimpleLSTM, SimpleRN


DEFAULT_METADATA_DIR = (
    Path(__file__).resolve().parents[1] / "metadata"
)


@dataclass(frozen=True)
class Run:
    """Fully resolved run specification."""

    experiment_id: str
    checkpoint_name: str
    model_type: str
    paper_label: str
    split: int
    model_index: int
    concat_entity_vector: bool
    output_dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int | None
    seed: int


def lookup_run(
    *,
    experiment_id: str,
    model_type: str,
    entity_condition: str,
    split: int,
    model_index: int,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
) -> Run:
    """Look up one run from the experiment and seed metadata."""

    metadata_root = Path(metadata_dir)
    experiments = _read_json(metadata_root / "experiments.json")
    checkpoint_seeds = _read_json(metadata_root / "checkpoint_seeds.json")

    experiment_entry = _find_experiment_entry(experiments, experiment_id)
    entity_entry = _find_entity_entry(experiment_entry, entity_condition)
    family_entry = _find_family_entry(experiment_entry, model_type)

    checkpoint_name = (
        family_entry["checkpoint_name_prefix_template"].format(
            entity_code=entity_entry["entity_code"]
        )
        + experiment_entry["checkpoint_name_suffix_template"].format(
            split=split,
            model_index=model_index,
        )
    )
    expected_suffix = "_best_model.pt"
    if not checkpoint_name.endswith(expected_suffix):
        raise ValueError(
            f"Checkpoint name {checkpoint_name} is missing expected suffix {expected_suffix}"
        )

    checkpoint_stem = checkpoint_name.removesuffix(expected_suffix)

    seed_entry = checkpoint_seeds["experiments"][experiment_id][entity_entry["label"]]
    if checkpoint_stem not in seed_entry["checkpoint_seeds"]:
        raise ValueError(f"Missing seed for checkpoint {checkpoint_stem}")

    return Run(
        experiment_id=experiment_id,
        checkpoint_name=checkpoint_name,
        model_type=family_entry["model_type"],
        paper_label=family_entry["paper_label"],
        split=split,
        model_index=model_index,
        concat_entity_vector=entity_entry["concat_entity_vector"],
        output_dim=entity_entry["output_dim"],
        hidden_dim=family_entry["hidden_dim"],
        n_layers=family_entry["n_layers"],
        n_heads=family_entry.get("n_heads"),
        seed=int(seed_entry["checkpoint_seeds"][checkpoint_stem]),
    )


def build_model(
    run_spec: Run,
    *,
    vocab_size: int = len(VOCAB),
) -> torch.nn.Module:
    """Instantiate the model for one run."""

    if run_spec.model_type == "SIMPLE_RN":
        params = RecurrentParams(
            hidden_dim=run_spec.hidden_dim,
            n_layers=run_spec.n_layers,
        )
        return SimpleRN(vocab_size, run_spec.output_dim, params)

    if run_spec.model_type == "SIMPLE_LSTM":
        params = RecurrentParams(
            hidden_dim=run_spec.hidden_dim,
            n_layers=run_spec.n_layers,
        )
        return SimpleLSTM(vocab_size, run_spec.output_dim, params)

    if run_spec.model_type == "SIMPLE_GRU":
        params = RecurrentParams(
            hidden_dim=run_spec.hidden_dim,
            n_layers=run_spec.n_layers,
        )
        return SimpleGRU(vocab_size, run_spec.output_dim, params)

    if run_spec.model_type == "ABS_ATTN":
        params = AttentionParams(
            hidden_dim=run_spec.hidden_dim,
            n_layers=run_spec.n_layers,
            n_heads=_require_n_heads(run_spec),
            pe_type="sinusoidal",
        )
        return SimpleAttentionModel(vocab_size, run_spec.output_dim, params)

    if run_spec.model_type == "ROPE_ATTN":
        params = AttentionParams(
            hidden_dim=run_spec.hidden_dim,
            n_layers=run_spec.n_layers,
            n_heads=_require_n_heads(run_spec),
            pe_type="rope",
        )
        return SimpleAttentionModel(vocab_size, run_spec.output_dim, params)

    raise ValueError(f"Unsupported model type: {run_spec.model_type}")


def _require_n_heads(run_spec: Run) -> int:
    if run_spec.n_heads is None:
        raise ValueError(f"Run spec for {run_spec.model_type} is missing n_heads")
    return run_spec.n_heads


def _find_experiment_entry(experiments: dict, experiment_id: str) -> dict:
    for experiment_entry in experiments["experiments"]:
        if experiment_entry["experiment_id"] == experiment_id:
            return experiment_entry
    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def _find_entity_entry(experiment_entry: dict, entity_condition: str) -> dict:
    for entity_entry in experiment_entry["entity_conditions"]:
        if entity_condition in {
            entity_entry["label"],
            entity_entry["entity_code"],
            _entity_condition_label(entity_entry),
        }:
            return entity_entry
    raise ValueError(
        f"Unknown entity condition {entity_condition} for experiment {experiment_entry['experiment_id']}"
    )


def _entity_condition_label(entity_entry: dict) -> str:
    if entity_entry["concat_entity_vector"]:
        return "with_entity"
    return "no_entity"


def _find_family_entry(experiment_entry: dict, model_type: str) -> dict:
    for family_entry in experiment_entry["model_families"]:
        if family_entry["model_type"] == model_type:
            return family_entry
    raise ValueError(
        f"Unknown model type {model_type} for experiment {experiment_entry['experiment_id']}"
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load one saved checkpoint state."""

    return torch.load(
        Path(checkpoint_path),
        map_location=torch.device(device),
        weights_only=False,
    )


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load saved model weights into an already constructed model."""

    checkpoint_state = load_checkpoint_state(checkpoint_path, device=device)
    if "model_state" not in checkpoint_state:
        raise ValueError(f"Checkpoint does not contain model_state: {checkpoint_path}")

    target_device = torch.device(device)
    model.load_state_dict(checkpoint_state["model_state"])
    model.to(target_device)
    model.eval()
    return checkpoint_state


def run_from_checkpoint_state(checkpoint_state: dict[str, Any]) -> Run:
    """Build a run spec directly from the released checkpoint metadata."""

    required_fields = (
        "experiment_id",
        "file_name",
        "model_type",
        "paper_label",
        "split",
        "model_index",
        "concat_entity_vector",
        "output_dim",
        "hidden_dim",
        "n_layers",
        "n_heads",
        "seed",
    )
    missing_fields = [field for field in required_fields if field not in checkpoint_state]
    if missing_fields:
        raise ValueError(
            "Checkpoint does not contain enough metadata to resolve plot-training run identity: "
            + ", ".join(missing_fields)
        )

    return Run(
        experiment_id=str(checkpoint_state["experiment_id"]),
        checkpoint_name=str(checkpoint_state["file_name"]),
        model_type=str(checkpoint_state["model_type"]),
        paper_label=str(checkpoint_state["paper_label"]),
        split=int(checkpoint_state["split"]),
        model_index=int(checkpoint_state["model_index"]),
        concat_entity_vector=bool(checkpoint_state["concat_entity_vector"]),
        output_dim=int(checkpoint_state["output_dim"]),
        hidden_dim=int(checkpoint_state["hidden_dim"]),
        n_layers=int(checkpoint_state["n_layers"]),
        n_heads=int(checkpoint_state["n_heads"]) if checkpoint_state["n_heads"] is not None else None,
        seed=int(checkpoint_state["seed"]),
    )


def resolve_training_history(
    checkpoint_state: dict[str, Any],
    *,
    trajectory_root: Path,
) -> dict[str, list[float]]:
    """Resolve normalized training-history metrics from the released JSON trajectories."""

    experiment_id = checkpoint_state["experiment_id"]
    file_name = checkpoint_state["file_name"]
    if not experiment_id or not file_name:
        raise ValueError("Checkpoint state has empty experiment_id or file_name.")

    expected_suffix = "_best_model.pt"
    if not file_name.endswith(expected_suffix):
        raise ValueError(f"file_name must end with {expected_suffix!r}: {file_name}")

    json_name = file_name.removesuffix(expected_suffix) + ".json"
    json_path = trajectory_root / experiment_id / json_name

    if not json_path.exists():
        raise ValueError(f"Trajectory JSON not found: {json_path}")

    with open(json_path) as f:
        payload = json.load(f)

    source_history = payload["training_history"]
    if not isinstance(source_history, dict):
        raise ValueError(f"training_history is not a dict in: {json_path}")
    return _normalize_training_history(source_history)


def _normalize_training_history(history: dict[str, Any]) -> dict[str, list[float]]:
    normalized_history: dict[str, list[float]] = {}
    for key, values in history.items():
        if not isinstance(values, list):
            raise ValueError(f"training_history[{key!r}] is not a list")
        normalized_history[key] = [float(value) for value in values]
    if not normalized_history:
        raise ValueError("Training history is empty after normalization")
    return normalized_history
