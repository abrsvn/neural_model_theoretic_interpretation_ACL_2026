"""Model metadata for cross-model appendix analysis."""

from __future__ import annotations

import re

from data.dataset import SYSTEMATICITY_GROUPS
from evaluation.systematicity import GROUP_OUTPUT_KEYS


GROUP_KEYS = tuple(GROUP_OUTPUT_KEYS[group_name] for group_name in SYSTEMATICITY_GROUPS)
GROUP_DISPLAY = {
    "word_group": "Word",
    "sentence_group": "Sentence",
    "complex_event": "Complex Event",
    "basic_event": "Basic Event",
}
GROUP_COLORS = {
    "word_group": "#1f77b4",
    "sentence_group": "#ff7f0e",
    "complex_event": "#2ca02c",
    "basic_event": "#d62728",
}
COMPETING_EVENT_ORDER = tuple(GROUP_DISPLAY[group_key] for group_key in GROUP_KEYS)
COMPETING_EVENT_COLORS = {
    GROUP_DISPLAY[group_key]: GROUP_COLORS[group_key]
    for group_key in GROUP_KEYS
}
SPLITS = ("S1", "S2")

_ARCHITECTURE_ORDER = {
    "SRN": 0,
    "GRU": 1,
    "LSTM": 2,
    "Attn": 3,
}
_RECURRENT_PATTERN = re.compile(
    r"^(?P<family>srn|lstm|gru)_l(?P<n_layers>\d+)_h(?P<hidden_dim>\d+)_"
    r"nosem_(?P<entity_code>noent|went)_s(?P<split>[12])_m(?P<model_index>\d+)$"
)
_ATTENTION_PATTERN = re.compile(
    r"^attn_(?P<pe_type>abspe|rope)_l(?P<n_layers>\d+)_h(?P<hidden_dim>\d+)_"
    r"nh(?P<n_heads>\d+)_nosem_(?P<entity_code>noent|went)_s(?P<split>[12])_m(?P<model_index>\d+)$"
)
_ARCH_FAMILY_BY_MODEL_TYPE = {
    "SIMPLE_RN": "SRN",
    "SIMPLE_LSTM": "LSTM",
    "SIMPLE_GRU": "GRU",
    "ABS_ATTN": "Attn_AbsPE",
    "ROPE_ATTN": "Attn_RoPE",
}


def parse_checkpoint_name(model_name: str) -> dict[str, int | str]:
    """Parse one checkpoint stem into cross-model fields."""

    recurrent_match = _RECURRENT_PATTERN.fullmatch(model_name)
    if recurrent_match is not None:
        fields = recurrent_match.groupdict()
        family = fields["family"]
        model_type = {
            "srn": "SIMPLE_RN",
            "lstm": "SIMPLE_LSTM",
            "gru": "SIMPLE_GRU",
        }[family]
        return {
            "model_type": model_type,
            "arch_family": _ARCH_FAMILY_BY_MODEL_TYPE[model_type],
            "hidden_dim": int(fields["hidden_dim"]),
            "n_layers": int(fields["n_layers"]),
            "entity_code": fields["entity_code"],
            "split": int(fields["split"]),
            "model_index": int(fields["model_index"]),
        }

    attention_match = _ATTENTION_PATTERN.fullmatch(model_name)
    if attention_match is None:
        raise ValueError(f"Unsupported model name: {model_name}")

    fields = attention_match.groupdict()
    model_type = {
        "abspe": "ABS_ATTN",
        "rope": "ROPE_ATTN",
    }[fields["pe_type"]]
    return {
        "model_type": model_type,
        "arch_family": _ARCH_FAMILY_BY_MODEL_TYPE[model_type],
        "hidden_dim": int(fields["hidden_dim"]),
        "n_layers": int(fields["n_layers"]),
        "n_heads": int(fields["n_heads"]),
        "entity_code": fields["entity_code"],
        "split": int(fields["split"]),
        "model_index": int(fields["model_index"]),
    }


def entity_label(entity_code: str) -> str:
    """Normalize filename entity codes for cross-model labels."""

    if entity_code == "noent":
        return "noent"
    if entity_code == "went":
        return "ent"
    raise ValueError(f"Unsupported entity code: {entity_code}")


def make_architecture_label(arch_family: str, hidden_dim: int, default_hidden_dim: int) -> str:
    """Return the cross-model architecture label, with H-suffix for variants."""

    if hidden_dim == default_hidden_dim:
        return arch_family
    return f"{arch_family}_H{hidden_dim}"


def sort_architectures(architecture_names: list[str]) -> list[str]:
    """Sort cross-model architecture labels in paper order."""

    return sorted(architecture_names, key=_architecture_sort_key)


def arch_display(architecture_name: str) -> str:
    """Convert underscore-delimited cross-model labels to display text."""

    return architecture_name.replace("_", " ")


def is_variant_architecture(architecture_name: str) -> bool:
    """Return whether an architecture label carries a hidden-dim variant suffix."""

    return bool(re.search(r"_H\d+$", architecture_name))


def _architecture_sort_key(architecture_name: str) -> tuple[int, int, str, str]:
    base_name = architecture_name
    variant_match = re.search(r"_H(?P<hidden_dim>\d+)$", architecture_name)
    if variant_match is not None:
        base_name = architecture_name[: variant_match.start()]
    family = base_name.split("_")[0]
    family_order = _ARCHITECTURE_ORDER.get(family)
    if family_order is None:
        raise ValueError(f"Unsupported cross-model architecture: {architecture_name}")

    is_variant = 1 if variant_match is not None else 0
    return (family_order, is_variant, base_name, architecture_name)
