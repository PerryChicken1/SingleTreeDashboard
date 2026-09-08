from __future__ import annotations

from pathlib import Path


_COMPONENT_DIR = Path(__file__).resolve().parent / "frontend" / "build"
_component = None


def _get_component():
	global _component
	if _component is None:
		import streamlit.components.v1 as components

		_component = components.declare_component(
			"thinning_schedule_chart",
			path=str(_COMPONENT_DIR),
		)
	return _component


def thinning_schedule_chart(
	n_thinnings: int,
	current_value: float,
	future_crop_value: float,
	unit_label: str,
	value_step: float = 1.0,
	default_targets: list[float] | None = None,
	key: str | None = None,
) -> list[float]:
	"""Render the thinning curve and return remaining targets."""
	n_thinnings = max(1, min(5, int(n_thinnings)))
	current_value = max(0.0, float(current_value))
	future_crop_value = max(0.0, min(current_value, float(future_crop_value)))
	value_step = max(0.001, float(value_step))
	defaults = list(default_targets or [])[:n_thinnings]
	if len(defaults) != n_thinnings:
		defaults = [current_value] * n_thinnings

	result = _get_component()(
		nThinnings=n_thinnings,
		currentValue=current_value,
		futureCropValue=future_crop_value,
		unitLabel=unit_label,
		valueStep=value_step,
		targets=defaults,
		default={"targets": defaults},
		key=key,
	)
	if not isinstance(result, dict) or not isinstance(result.get("targets"), list):
		return defaults
	return [float(value) for value in result["targets"][:n_thinnings]]
