"""Sentence-record loading from the included CSV files."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

MINIMUM_RELEASE_COLUMNS = (
    "sentence",
    "consistent",
    "tokens",
    "test_group",
    "systematicity_pattern",
    "systematicity_pattern_original",
    "complexity_level",
    "modifier_count",
    "described_events",
    "described_event_structure",
    "described_conjuncts",
    "competing_events",
    "has_toys",
)

SYSTEMATICITY_GROUPS = ("Word", "Sentence", "Complex_Event", "Basic_Event")


def parse_boollike(value: object) -> bool:
    """Normalize the boolean-like CSV fields used in the CSV files."""

    if value is None:
        return False

    text = str(value).strip().lower()
    if not text:
        return False
    return text in {"1", "true", "t", "yes", "y"}


def parse_pipe_separated(value: object, *, required: bool = False) -> list[str]:
    """Parse a pipe-separated field from the CSV files."""

    if value is None:
        if required:
            raise ValueError("Required pipe-separated field is missing")
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        if required:
            raise ValueError("Required pipe-separated field is empty")
        return []

    return [item.strip() for item in text.split("|") if item.strip()]


def normalize_optional_text(value: object) -> str | None:
    """Normalize optional string fields from the CSV files."""

    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


@dataclass(frozen=True)
class EventStructure:
    """Event-structure representation used in the CSV files."""

    event_type: str
    agent: str | set[str]
    theme: str | set[str] | None = None
    patient: str | set[str] | None = None
    location: str | set[str] | None = None
    manner: str | None = None

    @classmethod
    def from_string(cls, text: str) -> EventStructure:
        """Parse the exact repr format stored in the CSV files."""

        parsed = ast.parse(text, mode="eval")
        expr = parsed.body

        if not isinstance(expr, ast.Call):
            raise ValueError(f"Invalid EventStructure string: {text}")
        if not isinstance(expr.func, ast.Name) or expr.func.id != "EventStructure":
            raise ValueError(f"Invalid EventStructure constructor: {text}")
        if expr.args:
            raise ValueError(f"EventStructure must use keyword arguments only: {text}")

        kwargs: dict[str, object] = {}
        for keyword in expr.keywords:
            if keyword.arg is None:
                raise ValueError(f"EventStructure does not support **kwargs: {text}")
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)

        return cls(**kwargs)


@dataclass(frozen=True)
class SentenceRecord:
    """One sentence row with the sentence metadata and structure fields used here."""

    sentence: str
    consistent: bool
    tokens: tuple[str, ...]
    test_group: str | None
    systematicity_pattern: str | None
    systematicity_pattern_original: bool
    complexity_level: str | None
    modifier_count: int | None
    described_events: str
    described_event_structure: str
    described_conjuncts: tuple[str, ...]
    competing_events: tuple[str, ...]
    has_toys: bool

    def event_structure(self) -> EventStructure:
        """Parse the event-structure repr on demand."""

        return EventStructure.from_string(self.described_event_structure)


def load_sentence_records(
    csv_path: str | Path,
    *,
    consistent_only: bool = False,
) -> list[SentenceRecord]:
    """Load one CSV file into sentence records."""

    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        missing_columns = [
            column for column in MINIMUM_RELEASE_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"CSV file {path} is missing required columns: {missing_columns}"
            )

        rows: list[SentenceRecord] = []
        for raw_row in reader:
            consistent = parse_boollike(raw_row["consistent"])
            if consistent_only and not consistent:
                continue

            modifier_count_text = normalize_optional_text(raw_row["modifier_count"])
            modifier_count = None if modifier_count_text is None else int(modifier_count_text)

            rows.append(
                SentenceRecord(
                    sentence=raw_row["sentence"].strip(),
                    consistent=consistent,
                    tokens=tuple(parse_pipe_separated(raw_row["tokens"], required=True)),
                    test_group=normalize_optional_text(raw_row["test_group"]),
                    systematicity_pattern=normalize_optional_text(raw_row["systematicity_pattern"]),
                    systematicity_pattern_original=parse_boollike(
                        raw_row["systematicity_pattern_original"]
                    ),
                    complexity_level=normalize_optional_text(raw_row["complexity_level"]),
                    modifier_count=modifier_count,
                    described_events=raw_row["described_events"].strip(),
                    described_event_structure=raw_row["described_event_structure"].strip(),
                    described_conjuncts=tuple(parse_pipe_separated(raw_row["described_conjuncts"])),
                    competing_events=tuple(parse_pipe_separated(raw_row["competing_events"])),
                    has_toys=parse_boollike(raw_row["has_toys"]),
                )
            )

    return rows


def consistent_records(records: list[SentenceRecord]) -> list[SentenceRecord]:
    """Return the consistent subset used by the evaluation code."""

    return [record for record in records if record.consistent]
