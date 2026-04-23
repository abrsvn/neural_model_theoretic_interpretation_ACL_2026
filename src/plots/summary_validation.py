"""Shared ambiguity checks for paper summary and history rows."""

from __future__ import annotations


def require_unambiguous_summary_rows(summary_rows: list[dict[str, object]]) -> None:
    """
    Ambiguity Guard:
    Prevents silent merging of experiments that share model_type identifiers (e.g. ABS_ATTN)
    but differ in capacity (hidden_dim) or other architectural properties.
    When a user passes CSV files from multiple experiments without filtering via --experiment-id,
    this guard prevents generating corrupted, averaged plots or tables.
    """

    signatures_by_key: dict[tuple[str, str, str], set[tuple[str, str, str, str, str]]] = {}
    for row in summary_rows:
        key = (
            str(row["model_type"]),
            str(row["entity_condition"]),
            str(row["group_name"]),
        )
        signatures_by_key.setdefault(key, set()).add(summary_run_signature(row))

    _raise_on_ambiguous_signatures(
        signatures_by_key,
        "Mixed experiment rows require --experiment-id. Ambiguous summary rows:",
    )


def require_unambiguous_history_records(history_records: list[dict[str, object]]) -> None:
    signatures_by_key: dict[tuple[str, str], set[tuple[str, str, str, str, str]]] = {}
    for record in history_records:
        key = (
            str(record["model_type"]),
            str(record["entity_condition"]),
        )
        signatures_by_key.setdefault(key, set()).add(summary_run_signature(record))

    _raise_on_ambiguous_signatures(
        signatures_by_key,
        "Mixed experiment records require --experiment-id. Ambiguous training-history records:",
    )


def summary_run_signature(
    row: dict[str, object],
) -> tuple[str, str, str, str, str]:
    return (
        str(row["experiment_id"]),
        str(row["paper_label"]),
        str(row["hidden_dim"]),
        str(row["n_layers"]),
        str(row["n_heads"]),
    )


def format_summary_run_signature(
    signature: tuple[str, str, str, str, str],
) -> str:
    experiment_id, paper_label, hidden_dim, n_layers, n_heads = signature
    return (
        f"experiment_id={experiment_id}, paper_label={paper_label}, "
        f"hidden_dim={hidden_dim}, n_layers={n_layers}, n_heads={n_heads}"
    )


def _raise_on_ambiguous_signatures(
    signatures_by_key: dict[tuple[str, ...], set[tuple[str, str, str, str, str]]],
    prefix: str,
) -> None:
    problems = []
    for key in sorted(signatures_by_key):
        signatures = signatures_by_key[key]
        if len(signatures) < 2:
            continue
        signatures_text = "; ".join(
            format_summary_run_signature(signature)
            for signature in sorted(signatures)
        )
        problems.append(f"{' / '.join(key)} -> {signatures_text}")

    if problems:
        raise ValueError(prefix + "\n" + "\n".join(problems))
