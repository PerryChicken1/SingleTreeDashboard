from __future__ import annotations

from pathlib import Path
from typing import Iterable


_COMPONENT_DIR = Path(__file__).resolve().parent / "frontend" / "build"
_component = None


def _get_component():
	global _component
	if _component is None:
		import streamlit.components.v1 as components

		_component = components.declare_component(
			"objective_weight_slider",
			path=str(_COMPONENT_DIR),
		)
	return _component


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
	return max(minimum, min(maximum, float(value)))


def markers_to_weights(markers: Iterable[float], count: int) -> list[float]:
	"""Convert sorted marker positions on [0, 1] into segment weights."""
	if count <= 0:
		return []
	if count == 1:
		return [1.0]

	clean_markers = sorted(_clamp(marker) for marker in list(markers)[: count - 1])
	if len(clean_markers) < count - 1:
		clean_markers.extend((index + 1) / count for index in range(len(clean_markers), count - 1))
	clean_markers = sorted(clean_markers[: count - 1])

	points = [0.0, *clean_markers, 1.0]
	weights = [round(points[index + 1] - points[index], 6) for index in range(count)]
	if weights:
		weights[-1] = round(1.0 - sum(weights[:-1]), 6)
	return weights


def weights_to_markers(weights: Iterable[float], count: int) -> list[float]:
	"""Convert objective weights into cumulative marker positions."""
	if count <= 1:
		return []

	clean_weights = [max(0.0, float(weight)) for weight in list(weights)[:count]]
	if len(clean_weights) != count or sum(clean_weights) <= 0:
		clean_weights = [1.0 / count] * count
	else:
		total = sum(clean_weights)
		clean_weights = [weight / total for weight in clean_weights]

	markers = []
	running_total = 0.0
	for weight in clean_weights[:-1]:
		running_total += weight
		markers.append(round(_clamp(running_total), 6))
	return markers


def objective_weight_slider(
	objectives: list[str],
	labels: dict[str, str] | None = None,
	default_weights: dict[str, float] | None = None,
	key: str | None = None,
	min_weight: float = 0.0,
	step: float = 0.001,
	colors: list[str] | None = None,
	value_scale: float = 1.0,
	value_suffix: str = "",
	show_percentage: bool = False,
	show_equal_button: bool = True,
	boundary_band: float = 0.0,
	boundary_colors: list[str] | None = None,
) -> dict[str, float]:
	"""Render a multi-marker Streamlit slider and return weights by objective key."""
	if not objectives:
		return {}
	if len(objectives) == 1:
		return {objectives[0]: 1.0}

	labels = labels or {}
	weights = [
		float(default_weights.get(objective, 1.0 / len(objectives)))
		if default_weights
		else 1.0 / len(objectives)
		for objective in objectives
	]
	markers = weights_to_markers(weights, len(objectives))
	normalized_weights = markers_to_weights(markers, len(objectives))
	default = {
		"markers": markers,
		"weights": dict(zip(objectives, normalized_weights)),
	}

	result = _get_component()(
		objectives=objectives,
		labels=[labels.get(objective, objective) for objective in objectives],
		markers=markers,
		minWeight=_clamp(min_weight),
		step=max(float(step), 0.000001),
		colors=colors,
		valueScale=float(value_scale),
		valueSuffix=value_suffix,
		showPercentage=show_percentage,
		showEqualButton=show_equal_button,
		boundaryBand=_clamp(boundary_band),
		boundaryColors=boundary_colors,
		default=default,
		key=key,
	)

	if not isinstance(result, dict):
		result = default

	result_weights = result.get("weights")
	if isinstance(result_weights, dict):
		values = [float(result_weights.get(objective, 0.0)) for objective in objectives]
	else:
		values = markers_to_weights(result.get("markers", markers), len(objectives))

	total = sum(values)
	if total <= 0:
		values = normalized_weights
	else:
		values = [round(value / total, 6) for value in values]
		values[-1] = round(1.0 - sum(values[:-1]), 6)

	return dict(zip(objectives, values))
