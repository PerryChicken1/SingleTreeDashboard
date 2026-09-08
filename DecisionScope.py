"""Shared semantics for the two complementary sides of a binary decision."""

from enum import StrEnum


class DecisionScope(StrEnum):
    """Which trees an objective or constraint evaluates.

    ``SELECT`` represents entries where the canonical decision vector is one;
    ``NOT_SELECT`` represents its complement.  Treatment-specific optimisers
    supply the human meaning of those sets (for example, cut versus leave).
    """

    SELECT = "select"
    NOT_SELECT = "not_select"


def coerce_decision_scope(value: DecisionScope | str) -> DecisionScope:
    """Return a validated decision scope with a useful error message."""
    try:
        return DecisionScope(value)
    except ValueError as exc:
        choices = ", ".join(scope.value for scope in DecisionScope)
        raise ValueError(f"applies_to must be one of: {choices}.") from exc
