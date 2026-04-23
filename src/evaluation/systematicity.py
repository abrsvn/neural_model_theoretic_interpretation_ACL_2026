"""Systematicity evaluation and result export."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.dataset import SYSTEMATICITY_GROUPS, SentenceRecord, consistent_records
from data.targets import TargetBuilder
from .batching import encode_tokens, select_final_outputs
from .metrics import compute_comprehension_score


logger = logging.getLogger(__name__)


GROUP_OUTPUT_KEYS = {
    "Word": "word_group",
    "Sentence": "sentence_group",
    "Complex_Event": "complex_event",
    "Basic_Event": "basic_event",
}


@dataclass(frozen=True)
class SystematicityGroupResult:
    """Per-group systematicity evaluation result."""

    group_name: str
    train_rows: list[dict[str, object]]
    test_rows: list[dict[str, object]]
    summary: dict[str, float]


def evaluate_systematicity_suite(
    model: torch.nn.Module,
    *,
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    target_builder: TargetBuilder,
    device: str | torch.device = "cpu",
    truth_only: bool = True,
) -> dict[str, SystematicityGroupResult]:
    """Evaluate the four systematicity groups for one trained model."""

    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    results = {
        "Word": evaluate_word_group(
            model,
            train_records=train_records,
            test_records=test_records,
            target_builder=target_builder,
            device=device_obj,
            truth_only=truth_only,
        ),
        "Sentence": evaluate_sentence_group(
            model,
            train_records=train_records,
            test_records=test_records,
            target_builder=target_builder,
            device=device_obj,
            truth_only=truth_only,
        ),
        "Complex_Event": evaluate_complex_event_group(
            model,
            train_records=train_records,
            test_records=test_records,
            target_builder=target_builder,
            device=device_obj,
            truth_only=truth_only,
        ),
        "Basic_Event": evaluate_basic_event_group(
            model,
            train_records=train_records,
            test_records=test_records,
            target_builder=target_builder,
            device=device_obj,
            truth_only=truth_only,
        ),
    }
    return results


def evaluate_word_group(
    model: torch.nn.Module,
    *,
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    target_builder: TargetBuilder,
    device: str | torch.device = "cpu",
    truth_only: bool = True,
) -> SystematicityGroupResult:
    """Evaluate the Word systematicity group."""

    train_group = _select_pattern_records(train_records, "Word")
    test_group = _select_pattern_records(test_records, "Word")
    _validate_no_overlap(train_group, test_group, group_name="Word")

    train_rows = [
        _score_word_record(model, record, target_builder, device=device, truth_only=truth_only)
        for record in train_group
    ]
    test_rows = [
        _score_word_record(model, record, target_builder, device=device, truth_only=truth_only)
        for record in test_group
    ]
    return SystematicityGroupResult(
        group_name="Word",
        train_rows=train_rows,
        test_rows=test_rows,
        summary={
            **_summarize_described_competing(train_rows, prefix="train"),
            **_summarize_described_competing(test_rows, prefix="test"),
            **_summarize_theoretical_bounds(test_rows, prefix=""),
            **_summarize_theoretical_bounds(train_rows, prefix="train"),
        },
    )


def evaluate_sentence_group(
    model: torch.nn.Module,
    *,
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    target_builder: TargetBuilder,
    device: str | torch.device = "cpu",
    truth_only: bool = True,
) -> SystematicityGroupResult:
    """Evaluate the Sentence systematicity group."""

    train_group = _select_pattern_records(train_records, "Sentence")
    test_group = _select_pattern_records(test_records, "Sentence")
    _validate_no_overlap(train_group, test_group, group_name="Sentence")

    train_rows = [
        _score_sentence_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
            compute_route_preference=False,
        )
        for record in train_group
    ]
    test_rows = [
        _score_sentence_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
            compute_route_preference=True,
        )
        for record in test_group
    ]

    summary = {
        **_summarize_sentence_rows(train_rows, prefix="train"),
        **_summarize_sentence_rows(test_rows, prefix="test"),
        **_summarize_theoretical_bounds(test_rows, prefix=""),
        **_summarize_theoretical_bounds(train_rows, prefix="train"),
    }
    route_preferences = [
        float(row["route_preference"])
        for row in test_rows
        if "route_preference" in row
    ]
    summary["avg_route_preference"] = _mean(route_preferences)

    return SystematicityGroupResult(
        group_name="Sentence",
        train_rows=train_rows,
        test_rows=test_rows,
        summary=summary,
    )


def evaluate_complex_event_group(
    model: torch.nn.Module,
    *,
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    target_builder: TargetBuilder,
    device: str | torch.device = "cpu",
    truth_only: bool = True,
) -> SystematicityGroupResult:
    """Evaluate the Complex_Event systematicity group."""

    train_group = _select_pattern_records(train_records, "Complex_Event")
    test_group = _select_pattern_records(test_records, "Complex_Event")
    _validate_no_overlap(train_group, test_group, group_name="Complex_Event")

    train_rows = [
        _score_complex_event_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
        )
        for record in train_group
    ]
    test_rows = [
        _score_complex_event_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
        )
        for record in test_group
    ]

    return SystematicityGroupResult(
        group_name="Complex_Event",
        train_rows=train_rows,
        test_rows=test_rows,
        summary={
            **_summarize_complex_event_rows(train_rows, prefix="train"),
            **_summarize_complex_event_rows(test_rows, prefix="test"),
            **_summarize_theoretical_bounds(test_rows, prefix=""),
            **_summarize_theoretical_bounds(train_rows, prefix="train"),
        },
    )


def evaluate_basic_event_group(
    model: torch.nn.Module,
    *,
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    target_builder: TargetBuilder,
    device: str | torch.device = "cpu",
    truth_only: bool = True,
) -> SystematicityGroupResult:
    """Evaluate the Basic_Event systematicity group."""

    train_group = _select_pattern_records(train_records, "Basic_Event")
    test_group = _select_pattern_records(test_records, "Basic_Event")
    _validate_no_overlap(train_group, test_group, group_name="Basic_Event")

    train_rows = [
        _score_basic_event_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
        )
        for record in train_group
    ]
    test_rows = [
        _score_basic_event_record(
            model,
            record,
            target_builder,
            device=device,
            truth_only=truth_only,
        )
        for record in test_group
    ]

    summary = {
        **_summarize_basic_event_rows(train_rows, prefix="train"),
        **_summarize_basic_event_rows(test_rows, prefix="test"),
        **_summarize_theoretical_bounds(test_rows, prefix=""),
        **_summarize_theoretical_bounds(train_rows, prefix="train"),
    }
    summary["test_passes_threshold"] = float(summary["avg_test_described"] > 0.5)

    return SystematicityGroupResult(
        group_name="Basic_Event",
        train_rows=train_rows,
        test_rows=test_rows,
        summary=summary,
    )


def flatten_systematicity_results(
    results: dict[str, SystematicityGroupResult],
    *,
    canonical_only: bool = False,
) -> list[dict[str, object]]:
    """Flatten suite results into one row per sentence for CSV export."""

    flattened_rows: list[dict[str, object]] = []
    for group_name in SYSTEMATICITY_GROUPS:
        group_result = results[group_name]
        for split_name, rows in (("train", group_result.train_rows), ("test", group_result.test_rows)):
            for row in _filter_rows(rows, canonical_only=canonical_only):
                serialized_row = {
                    "group_name": group_name,
                    "output_key": GROUP_OUTPUT_KEYS[group_name],
                    "split": split_name,
                }
                for key, value in row.items():
                    serialized_row[key] = _serialize_csv_value(value)
                flattened_rows.append(serialized_row)
    return flattened_rows


def summarize_systematicity_results(
    results: dict[str, SystematicityGroupResult],
) -> list[dict[str, object]]:
    """Flatten suite summaries into one row per systematicity group."""

    summary_rows: list[dict[str, object]] = []
    for group_name in SYSTEMATICITY_GROUPS:
        group_result = results[group_name]
        summary_rows.append(
            {
                "group_name": group_name,
                "output_key": GROUP_OUTPUT_KEYS[group_name],
                "n_train": len(group_result.train_rows),
                "n_test": len(group_result.test_rows),
                "train_advantage": float(group_result.summary["avg_train_advantage"]),
                "test_advantage": float(group_result.summary["avg_test_advantage"]),
                "generalization_gap": float(
                    group_result.summary["avg_train_advantage"]
                    - group_result.summary["avg_test_advantage"]
                ),
                **group_result.summary,
            }
        )
    return summary_rows


def write_systematicity_rows_csv(
    results: dict[str, SystematicityGroupResult],
    output_path: str | Path,
    *,
    canonical_only: bool = False,
) -> None:
    """Write row-level systematicity results to CSV."""

    _write_csv(
        flatten_systematicity_results(results, canonical_only=canonical_only),
        output_path,
    )


def write_systematicity_summary_csv(
    results: dict[str, SystematicityGroupResult],
    output_path: str | Path,
) -> None:
    """Write aggregate systematicity summaries to CSV."""

    _write_csv(summarize_systematicity_results(results), output_path)


def _score_word_record(
    model: torch.nn.Module,
    record: SentenceRecord,
    target_builder: TargetBuilder,
    *,
    device: str | torch.device,
    truth_only: bool,
) -> dict[str, object]:
    described_formulas = list(record.described_conjuncts)
    if record.described_events not in described_formulas:
        described_formulas.append(record.described_events)

    row = _score_record_with_metadata(
        model,
        record,
        described_formulas=described_formulas,
        competing_formulas=list(record.competing_events),
        target_builder=target_builder,
        device=device,
        truth_only=truth_only,
    )
    row["described_score"] = float(row["described_scores"][-1])
    return row


def _score_sentence_record(
    model: torch.nn.Module,
    record: SentenceRecord,
    target_builder: TargetBuilder,
    *,
    device: str | torch.device,
    truth_only: bool,
    compute_route_preference: bool,
) -> dict[str, object]:
    components = _build_sentence_components(record)
    row = _score_record_with_metadata(
        model,
        record,
        described_formulas=components["ordered_formulas"],
        competing_formulas=list(record.competing_events),
        target_builder=target_builder,
        device=device,
        truth_only=truth_only,
    )

    described_scores = row["described_scores"]
    row["win_score"] = described_scores[0]
    row["lose_score"] = described_scores[1]

    next_index = 2
    component_scores = [float(row["win_score"]), float(row["lose_score"])]
    if components["location_formula"] is not None:
        row["location_score"] = described_scores[next_index]
        component_scores.append(float(row["location_score"]))
        next_index += 1
    if components["manner_formula"] is not None:
        row["manner_score"] = described_scores[next_index]
        component_scores.append(float(row["manner_score"]))
        next_index += 1

    full_formula_score = float(described_scores[-1])
    row["composition_preference"] = full_formula_score - float(np.mean(component_scores))
    if compute_route_preference and components["is_canonical"]:
        row["route_preference"] = row["composition_preference"]

    row["described_score"] = full_formula_score
    return row


def _score_complex_event_record(
    model: torch.nn.Module,
    record: SentenceRecord,
    target_builder: TargetBuilder,
    *,
    device: str | torch.device,
    truth_only: bool,
) -> dict[str, object]:
    components = _build_complex_event_components(record)
    row = _score_record_with_metadata(
        model,
        record,
        described_formulas=components["ordered_formulas"],
        competing_formulas=list(record.competing_events),
        target_builder=target_builder,
        device=device,
        truth_only=truth_only,
    )

    described_scores = row["described_scores"]
    row["play_score"] = described_scores[0]
    row["location_score"] = described_scores[1]

    component_scores = [float(row["play_score"]), float(row["location_score"])]
    if components["manner_formula"] is not None:
        row["manner_score"] = described_scores[2]
        component_scores.append(float(row["manner_score"]))

    row["described_score"] = described_scores[-1]
    row["composition_preference"] = float(row["described_score"]) - float(np.mean(component_scores))
    return row


def _score_basic_event_record(
    model: torch.nn.Module,
    record: SentenceRecord,
    target_builder: TargetBuilder,
    *,
    device: str | torch.device,
    truth_only: bool,
) -> dict[str, object]:
    components = _build_basic_event_components(record)
    row = _score_record_with_metadata(
        model,
        record,
        described_formulas=components["ordered_formulas"],
        competing_formulas=list(record.competing_events),
        target_builder=target_builder,
        device=device,
        truth_only=truth_only,
    )

    described_scores = row["described_scores"]
    row["play_toy_score"] = described_scores[0]
    row["described_score"] = described_scores[-1]

    if components["location_formula"] is not None:
        row["location_score"] = described_scores[1]
        row["composition_preference"] = float(row["described_score"]) - float(
            np.mean([float(row["play_toy_score"]), float(row["location_score"])])
        )
    return row


def _score_record_with_metadata(
    model: torch.nn.Module,
    record: SentenceRecord,
    *,
    described_formulas: list[str],
    competing_formulas: list[str],
    target_builder: TargetBuilder,
    device: str | torch.device,
    truth_only: bool,
) -> dict[str, object]:
    sentence_output = _compute_sentence_output(model, record.tokens, device=device)
    described_targets = [target_builder.build_target(formula) for formula in described_formulas]
    competing_targets = [target_builder.build_target(formula) for formula in competing_formulas]

    described_scores = _score_output_against_targets(
        sentence_output,
        described_targets,
        device=device,
        truth_only=truth_only,
    )
    competing_scores = _score_output_against_targets(
        sentence_output,
        competing_targets,
        device=device,
        truth_only=truth_only,
    )
    theoretical_max, theoretical_min = _compute_theoretical_bounds(
        record,
        target_builder,
        truth_only=truth_only,
        device=device,
    )

    return {
        "sentence": record.sentence,
        "tokens": list(record.tokens),
        "described_formulas": described_formulas,
        "competing_formulas": competing_formulas,
        "described_scores": described_scores,
        "competing_scores": competing_scores,
        # Scorers in each of the 4 test groups replace described_score with the full-formula score.
        # competing_score remains the row-level mean over the competing formulas.
        "competing_score": _mean(competing_scores),
        "num_described_events": len(described_formulas),
        "num_competing_events": len(competing_formulas),
        "theoretical_max": theoretical_max,
        "theoretical_min": theoretical_min,
        "complexity_level": record.complexity_level,
        "modifier_count": record.modifier_count,
        "systematicity_pattern_original": record.systematicity_pattern_original,
    }


def _compute_sentence_output(
    model: torch.nn.Module,
    tokens: tuple[str, ...],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    sequence = encode_tokens(tokens).unsqueeze(0).to(device)
    sequence_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs, _ = model(sequence, sequence_lengths)
        final_outputs = select_final_outputs(outputs, sequence_lengths)

    sentence_output = final_outputs.squeeze(0)
    if sentence_output.ndim != 1:
        raise ValueError(f"Expected final model output to be 1D, found {tuple(sentence_output.shape)}")
    return sentence_output


def _score_output_against_targets(
    sentence_output: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    device: str | torch.device,
    truth_only: bool,
) -> list[float]:
    if not targets:
        return []

    target_batch = torch.stack(targets).to(device)
    output_batch = sentence_output.unsqueeze(0).expand(target_batch.shape[0], -1)
    scores = compute_comprehension_score(
        output_batch,
        target_batch,
        truth_only=truth_only,
    )
    return scores.reshape(-1).cpu().tolist()


def _compute_theoretical_bounds(
    record: SentenceRecord,
    target_builder: TargetBuilder,
    *,
    truth_only: bool,
    device: str | torch.device,
) -> tuple[float, float]:
    described_target = target_builder.build_target(record.described_events).to(device)
    competing_targets = [target_builder.build_target(formula).to(device) for formula in record.competing_events]
    if not competing_targets:
        return (float("nan"), float("nan"))

    described_batch = described_target.unsqueeze(0)
    theoretical_max = float(
        compute_comprehension_score(
            described_batch,
            described_batch,
            truth_only=truth_only,
        ).item()
    )

    competing_scores = [
        float(
            compute_comprehension_score(
                described_batch,
                target.unsqueeze(0),
                truth_only=truth_only,
            ).item()
        )
        for target in competing_targets
    ]
    return (theoretical_max, _mean(competing_scores))


def _select_pattern_records(
    records: list[SentenceRecord],
    pattern_name: str,
) -> list[SentenceRecord]:
    filtered_records = [
        record
        for record in consistent_records(records)
        if record.systematicity_pattern == pattern_name
        if record.competing_events
    ]
    if not filtered_records:
        raise ValueError(f"No consistent records found for systematicity pattern {pattern_name}")
    return filtered_records


def _validate_no_overlap(
    train_records: list[SentenceRecord],
    test_records: list[SentenceRecord],
    *,
    group_name: str,
) -> None:
    train_sentences = {record.sentence.lower() for record in train_records}
    test_sentences = {record.sentence.lower() for record in test_records}
    overlap = sorted(train_sentences & test_sentences)
    if overlap:
        raise ValueError(f"Train/test overlap detected in {group_name}: {overlap[:5]}")


def _build_sentence_components(record: SentenceRecord) -> dict[str, object]:
    event = record.event_structure()
    if event.event_type == "win":
        winner = _require_atomic_value(event.agent, field_name="agent", record=record)
        loser = _require_atomic_value(event.patient, field_name="patient", record=record)
    elif event.event_type == "lose":
        loser = _require_atomic_value(event.agent, field_name="agent", record=record)
        winner = _require_atomic_value(event.patient, field_name="patient", record=record)
    else:
        raise ValueError(
            f"Sentence record {record.sentence!r} has unexpected event_type={event.event_type!r}"
        )

    location_formula = None
    if event.location is not None:
        location_formula = _build_joint_location_formula(winner, loser, event.location)

    manner_formula = None
    if event.manner is not None:
        manner_formula = f"(win_manner {event.manner})"

    ordered_formulas = [f"(win {winner})", f"(lose {loser})"]
    if location_formula is not None:
        ordered_formulas.append(location_formula)
    if manner_formula is not None:
        ordered_formulas.append(manner_formula)
    ordered_formulas.append(record.described_events)

    return {
        "ordered_formulas": ordered_formulas,
        "location_formula": location_formula,
        "manner_formula": manner_formula,
        "is_canonical": location_formula is None and manner_formula is None,
    }


def _build_complex_event_components(record: SentenceRecord) -> dict[str, object]:
    event = record.event_structure()
    if event.event_type != "play":
        raise ValueError(
            f"Complex_Event record {record.sentence!r} has unexpected event_type={event.event_type!r}"
        )

    agent = _require_atomic_value(event.agent, field_name="agent", record=record)
    theme = _require_atomic_value(event.theme, field_name="theme", record=record)
    if event.location is None:
        raise ValueError(f"Complex_Event record {record.sentence!r} is missing location")

    location_formula = _build_single_agent_location_formula(agent, event.location)
    manner_formula = None
    if event.manner is not None:
        manner_formula = f"(play_manner {agent} {event.manner})"

    ordered_formulas = [f"(play_game {agent} {theme})", location_formula]
    if manner_formula is not None:
        ordered_formulas.append(manner_formula)
    ordered_formulas.append(record.described_events)

    return {
        "ordered_formulas": ordered_formulas,
        "manner_formula": manner_formula,
    }


def _build_basic_event_components(record: SentenceRecord) -> dict[str, object]:
    event = record.event_structure()
    if event.event_type != "play":
        raise ValueError(
            f"Basic_Event record {record.sentence!r} has unexpected event_type={event.event_type!r}"
        )

    agent = _require_atomic_value(event.agent, field_name="agent", record=record)
    theme = _require_atomic_value(event.theme, field_name="theme", record=record)
    location_formula = None
    if event.location is not None:
        location_formula = _build_single_agent_location_formula(agent, event.location)

    ordered_formulas = [f"(play_toy {agent} {theme})"]
    if location_formula is not None:
        ordered_formulas.append(location_formula)
    if record.described_events not in ordered_formulas:
        ordered_formulas.append(record.described_events)

    return {
        "ordered_formulas": ordered_formulas,
        "location_formula": location_formula,
    }


def _build_joint_location_formula(
    winner: str,
    loser: str,
    location: str | set[str],
) -> str:
    if isinstance(location, str):
        return f"(and (location {winner} {location}) (location {loser} {location}))"

    disjuncts = [
        f"(and (location {winner} {candidate}) (location {loser} {candidate}))"
        for candidate in sorted(location)
    ]
    return f"(or {' '.join(disjuncts)})"


def _build_single_agent_location_formula(
    agent: str,
    location: str | set[str],
) -> str:
    if isinstance(location, str):
        return f"(location {agent} {location})"

    disjuncts = [f"(location {agent} {candidate})" for candidate in sorted(location)]
    return f"(or {' '.join(disjuncts)})"


def _require_atomic_value(
    value: str | set[str] | None,
    *,
    field_name: str,
    record: SentenceRecord,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Record {record.sentence!r} has non-atomic {field_name}: {value!r}"
        )
    return value


def _summarize_described_competing(
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"avg_{prefix}_described": _mean([float(row["described_score"]) for row in rows]),
        f"avg_{prefix}_competing": _mean([float(row["competing_score"]) for row in rows]),
        f"avg_{prefix}_advantage": _mean(_sentence_advantages(rows)),
    }


def _summarize_sentence_rows(
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"avg_{prefix}_win_component": _mean(_numeric_column(rows, "win_score")),
        f"avg_{prefix}_lose_component": _mean(_numeric_column(rows, "lose_score")),
        f"avg_{prefix}_location_component": _mean(_numeric_column(rows, "location_score")),
        f"avg_{prefix}_manner_component": _mean(_numeric_column(rows, "manner_score")),
        f"avg_{prefix}_described": _mean(_numeric_column(rows, "described_score")),
        f"avg_{prefix}_competing": _mean(_numeric_column(rows, "competing_score")),
        f"avg_{prefix}_advantage": _mean(_sentence_advantages(rows)),
        f"avg_{prefix}_composition_preference": _mean(_numeric_column(rows, "composition_preference")),
    }


def _summarize_complex_event_rows(
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"avg_{prefix}_play_comp": _mean(_numeric_column(rows, "play_score")),
        f"avg_{prefix}_loc_comp": _mean(_numeric_column(rows, "location_score")),
        f"avg_{prefix}_manner_comp": _mean(_numeric_column(rows, "manner_score")),
        f"avg_{prefix}_described": _mean(_numeric_column(rows, "described_score")),
        f"avg_{prefix}_competing": _mean(_numeric_column(rows, "competing_score")),
        f"avg_{prefix}_advantage": _mean(_sentence_advantages(rows)),
        f"avg_{prefix}_composition_preference": _mean(_numeric_column(rows, "composition_preference")),
    }


def _summarize_basic_event_rows(
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"avg_{prefix}_play_toy_comp": _mean(_numeric_column(rows, "play_toy_score")),
        f"avg_{prefix}_location_comp": _mean(_numeric_column(rows, "location_score")),
        f"avg_{prefix}_described": _mean(_numeric_column(rows, "described_score")),
        f"avg_{prefix}_competing": _mean(_numeric_column(rows, "competing_score")),
        f"avg_{prefix}_advantage": _mean(_sentence_advantages(rows)),
        f"avg_{prefix}_composition_preference": _mean(_numeric_column(rows, "composition_preference")),
    }


def _summarize_theoretical_bounds(
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> dict[str, float]:
    prefix_text = f"{prefix}_" if prefix else ""
    return {
        f"avg_{prefix_text}theoretical_max": _mean(_numeric_column(rows, "theoretical_max")),
        f"avg_{prefix_text}theoretical_min": _mean(_numeric_column(rows, "theoretical_min")),
    }


def _sentence_advantages(
    rows: list[dict[str, object]],
) -> list[float]:
    return [
        float(row["described_score"]) - float(row["competing_score"])
        for row in rows
    ]


def _numeric_column(
    rows: list[dict[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        value = float(row[key])
        if np.isfinite(value):
            values.append(value)
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _serialize_csv_value(value: object) -> object:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def _filter_rows(
    rows: list[dict[str, object]],
    *,
    canonical_only: bool,
) -> list[dict[str, object]]:
    if not canonical_only:
        return rows
    return [row for row in rows if bool(row["systematicity_pattern_original"])]


def _write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred_fields = [
        "group_name",
        "output_key",
        "split",
        "sentence",
        "n_train",
        "n_test",
    ]
    other_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in preferred_fields
        }
    )
    fieldnames = [field for field in preferred_fields if any(field in row for row in rows)] + other_fields

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    logger.info(f"Wrote {path}")
