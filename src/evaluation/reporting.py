"""Flatten evaluation outputs into plot-ready records."""

from __future__ import annotations

from checkpoints import Run
from .systematicity import GROUP_OUTPUT_KEYS, SystematicityGroupResult


def entity_condition_label(concat_entity_vector: bool) -> str:
    """Return the standard entity-condition label."""

    if concat_entity_vector:
        return "with_entity"
    return "no_entity"


def build_systematicity_summary_rows(
    run_spec: Run,
    results: dict[str, SystematicityGroupResult],
) -> list[dict[str, object]]:
    """Build one plot-ready summary row per systematicity group for a run."""

    rows: list[dict[str, object]] = []
    entity_condition = entity_condition_label(run_spec.concat_entity_vector)
    for group_name, group_result in results.items():
        rows.append(
            {
                "experiment_id": run_spec.experiment_id,
                "checkpoint_name": run_spec.checkpoint_name,
                "paper_label": run_spec.paper_label,
                "model_type": run_spec.model_type,
                "group_name": group_name,
                "output_key": GROUP_OUTPUT_KEYS[group_name],
                "entity_condition": entity_condition,
                "concat_entity_vector": run_spec.concat_entity_vector,
                "split": run_spec.split,
                "model_index": run_spec.model_index,
                "seed": run_spec.seed,
                "hidden_dim": run_spec.hidden_dim,
                "n_layers": run_spec.n_layers,
                "n_heads": run_spec.n_heads,
                "output_dim": run_spec.output_dim,
                **group_result.summary,
            }
        )
    return rows


def build_training_history_record(
    run_spec: Run,
    history: dict[str, list[float]],
) -> dict[str, object]:
    """Build one plot-ready grouped-history record for a run."""

    return {
        "experiment_id": run_spec.experiment_id,
        "checkpoint_name": run_spec.checkpoint_name,
        "paper_label": run_spec.paper_label,
        "model_type": run_spec.model_type,
        "entity_condition": entity_condition_label(run_spec.concat_entity_vector),
        "concat_entity_vector": run_spec.concat_entity_vector,
        "split": run_spec.split,
        "model_index": run_spec.model_index,
        "seed": run_spec.seed,
        "hidden_dim": run_spec.hidden_dim,
        "n_layers": run_spec.n_layers,
        "n_heads": run_spec.n_heads,
        "output_dim": run_spec.output_dim,
        "history": history,
    }
