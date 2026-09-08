from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


_COMPONENT_DIR = Path(__file__).resolve().parent / "frontend" / "build"
_component = None


def _get_component():
	global _component
	if _component is None:
		import streamlit.components.v1 as components

		_component = components.declare_component(
			"frequency_histogram",
			path=str(_COMPONENT_DIR),
		)
	return _component


def _clean_counts(values: Iterable[float], upper_bounds: np.ndarray) -> list[int]:
	values = list(values)
	return [
		max(0, min(int(upper_bounds[index]), int(round(float(value)))))
		for index, value in enumerate(values[:len(upper_bounds)])
	]


def frequency_histogram(
	labels: Iterable[str],
	actual_counts: Iterable[int],
	default_desired_counts: Iterable[int] | None = None,
	default_tolerances: Iterable[int] | None = None,
	x_axis_title: str = "",
	key: str | None = None,
) -> dict[str, list]:
	"""Render desired frequencies and independent count bounds."""
	clean_labels = [str(label) for label in labels]
	actual = [max(0, int(count)) for count in actual_counts]
	if len(clean_labels) != len(actual) or not clean_labels:
		return {"desired_counts": [], "tolerances": []}
	defaults = _clean_counts(default_desired_counts or actual, np.asarray(actual))
	tolerance_defaults = [max(0, int(round(float(value)))) for value in list(default_tolerances or defaults)[:len(actual)]]
	if len(tolerance_defaults) != len(actual):
		tolerance_defaults = defaults
	result = _get_component()(
		labels=clean_labels,
		actualCounts=actual,
		desiredCounts=defaults,
		tolerances=tolerance_defaults,
		xAxisTitle=x_axis_title,
		default={"desiredCounts": defaults, "tolerances": tolerance_defaults},
		key=key,
	)
	if not isinstance(result, dict):
		return {"desired_counts": defaults, "tolerances": tolerance_defaults}
	result_tolerances = [
		max(0, int(round(float(value))))
		for value in list(result.get("tolerances", tolerance_defaults))[:len(actual)]
	]
	if len(result_tolerances) != len(actual):
		result_tolerances = tolerance_defaults
	return {
		"desired_counts": _clean_counts(result.get("desiredCounts", defaults), np.asarray(actual)),
		"tolerances": result_tolerances,
	}