"""Target construction from formulas and NPZ weight files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PEOPLE = ("charlie", "heidi", "sophia")
GAMES = ("chess", "hide_n_seek", "soccer")
TOYS = ("puzzle", "ball", "doll")
LOCATIONS = ("bathroom", "bedroom", "playground", "street")
PLAY_MANNERS = ("well", "badly")
WIN_MANNERS = ("with_ease", "with_difficulty")

ENTITY_ORDER = (
    *PEOPLE,
    *GAMES,
    *TOYS,
    *LOCATIONS,
    *PLAY_MANNERS,
    *WIN_MANNERS,
)

ENTITY_TO_INDEX = {name: index for index, name in enumerate(ENTITY_ORDER)}


@dataclass(frozen=True)
class PropositionSpec:
    """Representation of one proposition dimension."""

    index: int
    predicate: str
    arguments: tuple[str, ...]
    display_label: str


def _build_propositions() -> tuple[PropositionSpec, ...]:
    propositions: list[PropositionSpec] = []
    index = 0

    for person in PEOPLE:
        for game in GAMES:
            propositions.append(
                PropositionSpec(
                    index=index,
                    predicate="play_game",
                    arguments=(person, game),
                    display_label=f"play({person},{game})",
                )
            )
            index += 1

    for person in PEOPLE:
        for toy in TOYS:
            propositions.append(
                PropositionSpec(
                    index=index,
                    predicate="play_toy",
                    arguments=(person, toy),
                    display_label=f"play({person},{toy})",
                )
            )
            index += 1

    for predicate in ("win", "lose"):
        for person in PEOPLE:
            propositions.append(
                PropositionSpec(
                    index=index,
                    predicate=predicate,
                    arguments=(person,),
                    display_label=f"{predicate}({person})",
                )
            )
            index += 1

    for person in PEOPLE:
        for location in LOCATIONS:
            propositions.append(
                PropositionSpec(
                    index=index,
                    predicate="location",
                    arguments=(person, location),
                    display_label=f"location({person},{location})",
                )
            )
            index += 1

    for person in PEOPLE:
        for manner in PLAY_MANNERS:
            propositions.append(
                PropositionSpec(
                    index=index,
                    predicate="play_manner",
                    arguments=(person, manner),
                    display_label=f"play_manner({person},{manner})",
                )
            )
            index += 1

    for manner in WIN_MANNERS:
        propositions.append(
            PropositionSpec(
                index=index,
                predicate="win_manner",
                arguments=(manner,),
                display_label=f"win_manner({manner})",
            )
        )
        index += 1

    if index != 44:
        raise ValueError(f"Expected 44 proposition slots, built {index}")

    return tuple(propositions)


PROPOSITION_SPECS = _build_propositions()
ATOMIC_TO_INDEX = {
    (spec.predicate, spec.arguments): spec.index
    for spec in PROPOSITION_SPECS
}

PREDICATE_ENTITY_ROLES = {
    "play_game": ("person", "game"),
    "play_toy": ("person", "toy"),
    "win": ("person",),
    "lose": ("person",),
    "location": ("person", "location"),
    "play_manner": ("person", "manner"),
    "win_manner": ("manner",),
}

SUPPORTED_PREDICATES = frozenset(PREDICATE_ENTITY_ROLES)
SUPPORTED_PREDICATE_ARITIES = {
    predicate: len(roles)
    for predicate, roles in PREDICATE_ENTITY_ROLES.items()
}


def atomic_formula_to_index(predicate: str, arguments: tuple[str, ...]) -> int:
    """Return the fixed proposition index for one atomic formula."""

    key = (predicate, arguments)
    if key not in ATOMIC_TO_INDEX:
        raise ValueError(f"Unsupported atomic formula: {predicate}{arguments}")
    return ATOMIC_TO_INDEX[key]


def atomic_formula_entities(predicate: str, arguments: tuple[str, ...]) -> tuple[str, ...]:
    """Return the ordered entity names mentioned by one atomic formula."""

    if predicate not in PREDICATE_ENTITY_ROLES:
        raise ValueError(f"Unsupported predicate: {predicate}")

    for entity_name in arguments:
        if entity_name not in ENTITY_TO_INDEX:
            raise ValueError(f"Unsupported entity name: {entity_name}")

    return arguments


@dataclass(frozen=True)
class ConstantFormula:
    """Formula constant."""

    value: bool


@dataclass(frozen=True)
class AtomicFormula:
    """Atomic predicate formula."""

    predicate: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class ConnectiveFormula:
    """Logical connective formula."""

    operator: str
    children: tuple[Formula, ...]


Formula = ConstantFormula | AtomicFormula | ConnectiveFormula


def tokenize_sexpr(text: str) -> list[str]:
    """Tokenize a parenthesized S-expression string."""

    tokens: list[str] = []
    current: list[str] = []

    for char in text:
        if char in {"(", ")"}:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
            continue
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)

    if current:
        tokens.append("".join(current))

    return tokens


def parse_formula(text: str) -> Formula:
    """Parse one formula string."""

    text = text.strip()
    if not text or text.lower() == "nan":
        raise ValueError("Formula string is empty")

    tokens = tokenize_sexpr(text)
    formula, next_index = _parse_expr(tokens, 0)
    if next_index != len(tokens):
        raise ValueError(f"Unexpected trailing tokens in formula: {text}")
    return formula


def _parse_expr(tokens: list[str], start_index: int) -> tuple[Formula, int]:
    token = tokens[start_index]
    if token != "(":
        if token == "true":
            return ConstantFormula(True), start_index + 1
        if token == "false":
            return ConstantFormula(False), start_index + 1
        raise ValueError(f"Expected '(' or a boolean constant, found {token!r}")

    if start_index + 2 >= len(tokens):
        raise ValueError("Incomplete parenthesized expression")

    head = tokens[start_index + 1]
    cursor = start_index + 2

    if head in {"and", "or"}:
        children: list[Formula] = []
        while cursor < len(tokens) and tokens[cursor] != ")":
            child, cursor = _parse_expr(tokens, cursor)
            children.append(child)
        if cursor >= len(tokens):
            raise ValueError(f"Unclosed {head!r} expression")
        if not children:
            raise ValueError(f"{head!r} requires at least one child")
        return ConnectiveFormula(head, tuple(children)), cursor + 1

    if head == "not":
        child, cursor = _parse_expr(tokens, cursor)
        if cursor >= len(tokens) or tokens[cursor] != ")":
            raise ValueError("'not' must contain exactly one child")
        return ConnectiveFormula("not", (child,)), cursor + 1

    arguments: list[str] = []
    while cursor < len(tokens) and tokens[cursor] != ")":
        argument = tokens[cursor]
        if argument == "(":
            raise ValueError(f"Atomic formula {head!r} cannot contain nested expressions")
        arguments.append(argument)
        cursor += 1

    if cursor >= len(tokens):
        raise ValueError(f"Unclosed atomic formula for predicate {head!r}")
    if head not in SUPPORTED_PREDICATES:
        raise ValueError(f"Unsupported predicate: {head}")
    expected_arity = SUPPORTED_PREDICATE_ARITIES[head]
    if len(arguments) != expected_arity:
        raise ValueError(
            f"Predicate {head!r} expects {expected_arity} arguments, found {len(arguments)}"
        )

    return AtomicFormula(head, tuple(arguments)), cursor + 1


def logical_and(vectors: list[torch.Tensor]) -> torch.Tensor:
    """Conjunction with duplicate-row removal for target composition."""

    if not vectors:
        raise ValueError("logical_and received an empty list")

    stacked = torch.stack(vectors)
    unique_vectors = torch.unique(stacked, dim=0)
    if unique_vectors.shape[0] == 1:
        return unique_vectors[0]
    return torch.prod(unique_vectors, dim=0)


def logical_or(vectors: list[torch.Tensor]) -> torch.Tensor:
    """Disjunction via De Morgan for target composition."""

    if not vectors:
        raise ValueError("logical_or received an empty list")
    if len(vectors) == 1:
        return vectors[0]
    return 1 - logical_and([1 - vector for vector in vectors])


@dataclass(frozen=True)
class TargetWeights:
    """Loaded proposition and entity weights."""

    proposition_weights: torch.Tensor
    entity_weights: torch.Tensor | None

    @classmethod
    def from_paths(
        cls,
        proposition_weights_path: str | Path,
        entity_weights_path: str | Path | None = None,
    ) -> TargetWeights:
        """Load the NPZ weight files."""

        props_data = np.load(Path(proposition_weights_path), allow_pickle=True)
        proposition_weights = torch.from_numpy(props_data["weights"].copy()).to(torch.float32)

        if proposition_weights.ndim != 2 or proposition_weights.shape[1] != 44:
            raise ValueError(
                "Expected proposition weights with shape (truth_dim, 44), "
                f"found {tuple(proposition_weights.shape)}"
            )

        entity_weights = None
        if entity_weights_path is not None:
            entity_data = np.load(Path(entity_weights_path), allow_pickle=True)
            entity_weights = torch.from_numpy(entity_data["entity_vectors"].copy()).to(torch.float32)
            entity_names = tuple(entity_data["entity_names"].tolist())

            if entity_weights.ndim != 2 or entity_weights.shape[1] != len(ENTITY_ORDER):
                raise ValueError(
                    "Expected entity vectors with shape (entity_dim, 17), "
                    f"found {tuple(entity_weights.shape)}"
                )
            if entity_names != ENTITY_ORDER:
                raise ValueError(
                    f"Entity order in {entity_weights_path} does not match the expected schema"
                )

        return cls(
            proposition_weights=proposition_weights,
            entity_weights=entity_weights,
        )

    @property
    def truth_dim(self) -> int:
        """Return the truth-conditional dimensionality."""

        return int(self.proposition_weights.shape[0])


@dataclass
class TargetBuilder:
    """Build 150-dim or 300-dim targets from formulas."""

    weights: TargetWeights
    with_entity_vectors: bool = False

    def __post_init__(self) -> None:
        if self.with_entity_vectors and self.weights.entity_weights is None:
            raise ValueError("Entity vectors requested but no entity weights were loaded")

    @property
    def output_dim(self) -> int:
        """Return the output dimensionality exposed to the evaluation code."""

        if self.with_entity_vectors:
            return 2 * self.weights.truth_dim
        return self.weights.truth_dim

    def build_target(self, formula_text: str) -> torch.Tensor:
        """Parse one formula string and convert it to a target."""

        formula = parse_formula(formula_text)
        truth_vector = self._build_truth_vector(formula)

        if not self.with_entity_vectors:
            return self._validate_target(truth_vector)

        entity_vector = self._build_entity_vector(formula)
        return self._validate_target(torch.cat([truth_vector, entity_vector], dim=0))

    def _build_truth_vector(self, formula: Formula) -> torch.Tensor:
        if isinstance(formula, ConstantFormula):
            if formula.value:
                return torch.ones(self.weights.truth_dim, dtype=torch.float32)
            return torch.zeros(self.weights.truth_dim, dtype=torch.float32)

        if isinstance(formula, AtomicFormula):
            index = atomic_formula_to_index(formula.predicate, formula.arguments)
            return self.weights.proposition_weights[:, index].clone()

        if formula.operator == "and":
            child_vectors = [self._build_truth_vector(child) for child in formula.children]
            return logical_and(child_vectors)
        if formula.operator == "or":
            child_vectors = [self._build_truth_vector(child) for child in formula.children]
            return logical_or(child_vectors)
        if formula.operator == "not":
            return 1 - self._build_truth_vector(formula.children[0])

        raise ValueError(f"Unsupported logical operator: {formula.operator}")

    def _build_entity_vector(self, formula: Formula) -> torch.Tensor:
        zero_vector = torch.zeros(self.weights.truth_dim, dtype=torch.float32)

        if isinstance(formula, ConstantFormula):
            return zero_vector

        if isinstance(formula, AtomicFormula):
            entity_vectors = [
                self.weights.entity_weights[:, ENTITY_TO_INDEX[entity_name]].clone()
                for entity_name in atomic_formula_entities(formula.predicate, formula.arguments)
            ]
            if not entity_vectors:
                return zero_vector
            if len(entity_vectors) == 1:
                return entity_vectors[0]
            return logical_and(entity_vectors)

        if formula.operator == "not":
            return self._build_entity_vector(formula.children[0])

        child_vectors = [
            self._build_entity_vector(child)
            for child in formula.children
        ]
        non_zero_child_vectors = [
            vector for vector in child_vectors if torch.any(vector > 1e-6)
        ]

        if not non_zero_child_vectors:
            return zero_vector
        if len(non_zero_child_vectors) == 1:
            return non_zero_child_vectors[0]
        if formula.operator == "and":
            return logical_and(non_zero_child_vectors)
        if formula.operator == "or":
            return logical_or(non_zero_child_vectors)

        raise ValueError(f"Unsupported logical operator: {formula.operator}")

    def _validate_target(self, vector: torch.Tensor) -> torch.Tensor:
        """Match the target validation behavior used here."""

        expected_dim = self.output_dim
        if vector.ndim != 1 or vector.shape[0] != expected_dim:
            raise ValueError(
                f"Expected target shape ({expected_dim},), found {tuple(vector.shape)}"
            )

        min_value = float(torch.min(vector))
        max_value = float(torch.max(vector))
        if min_value < -1e-6 or max_value > 1 + 1e-6:
            raise ValueError(
                "Target vector values must stay within [0, 1] up to tolerance; "
                f"found min={min_value}, max={max_value}"
            )

        return torch.clamp(vector, min=0.0, max=1.0)
