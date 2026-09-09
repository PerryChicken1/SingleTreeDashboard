from __future__ import annotations

import base64
import json
import html
import hashlib
from importlib.util import find_spec
import sys
from io import BytesIO
from pathlib import Path
import traceback
from typing import Dict, Iterable, List, Optional

import folium
from folium.plugins import Draw
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import shapefile
import streamlit as st
from shapely import maximum_inscribed_circle
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

try:
	from streamlit_folium import st_folium
except Exception:
	st_folium = None

try:
	import streamlit.components.v1 as components
except Exception:
	components = None


TOOL_DIR = Path(__file__).resolve().parent
APP_DIR = TOOL_DIR / "app"
if str(TOOL_DIR) not in sys.path:
	sys.path.insert(0, str(TOOL_DIR))

from Constraints import *
from Constraints import WaterProtectionConstraint, RoadHarvestConstraint
from Objectives import *
from Optimiser import *
from SingleTreeAlgorithms import *
from SingleTreeDataset import *
from GurobiLicense import GurobiLicenseError, gurobi_wls_session, wls_credentials
from app.components.objective_weight_slider import objective_weight_slider
from app.components.thinning_schedule_chart import thinning_schedule_chart
from app.components.frequency_histogram import frequency_histogram




st.set_page_config(
	page_title="SingleTreeOpt Dashboard",
	page_icon="ðŸŒ²",
	layout="wide",
)

STATIC_DIR = APP_DIR / "static"


@st.cache_data
def load_static_text(path: Path) -> str:
	"""Load a static dashboard file once per Streamlit process."""
	return path.read_text(encoding="utf-8")


@st.cache_data
def load_dashboard_metadata() -> dict:
	"""Load static dashboard metadata once per Streamlit process."""
	return json.loads(load_static_text(STATIC_DIR / "metadata.json"))


DASHBOARD_METADATA = load_dashboard_metadata()
TEST_DATASETS = json.loads(load_static_text(STATIC_DIR / "test_datasets.json"))
IMAGE_DIR = STATIC_DIR / DASHBOARD_METADATA["image_dir"]
AVAILABLE_LP_SOLVERS = ["CBC"] + (["GUROBI"] if find_spec("gurobipy") else [])
PROBLEM_METADATA = DASHBOARD_METADATA["problems"]
OBJECTIVE_LABELS = DASHBOARD_METADATA["objectives"]
ALGORITHM_METADATA = {
	"linear_programming": {
		**DASHBOARD_METADATA["algorithms"]["linear_programming"],
		"class": LinearProgrammingAlgorithm,
	},
	"differential_evolution": {
		**DASHBOARD_METADATA["algorithms"]["differential_evolution"],
		"class": DifferentialEvolutionAlgorithm,
	},
	"dual_annealing": {
		**DASHBOARD_METADATA["algorithms"]["dual_annealing"],
		"class": SimulatedAnnealingAlgorithm,
	},
	"genetic_algorithm": {
		**DASHBOARD_METADATA["algorithms"]["genetic_algorithm"],
		"class": GeneticAlgorithm,
	},
	"reinforcement_learning": {
		**DASHBOARD_METADATA["algorithms"]["reinforcement_learning"],
		"class": None,
	},
}
SUPPORTED_PROBLEMS = {
	key for key, value in PROBLEM_METADATA.items() if value["supported"]
}
SUPPORTED_ALGORITHMS = {
	key for key, value in ALGORITHM_METADATA.items() if value["supported"]
}
SUPPORTED_OBJECTIVES = list(OBJECTIVE_LABELS)


def results_csv(dataset: SingleTreeDataset, results: dict) -> bytes:
	"""Export original input rows with positional optimisation decisions."""
	export = dataset.source_data.copy()
	column = "optimisation_decision"
	while column in export.columns:
		column += "_result"
	export[column] = dataset.validate_decision_vector(results["decision_vector"])
	return export.to_csv(index=False).encode("utf-8")


def render_results_download(dataset: SingleTreeDataset, results: dict) -> None:
	st.download_button(
		"Download results",
		data=results_csv(dataset, results),
		file_name="optimisation-results.csv",
		mime="text/csv",
		key="download_optimisation_results",
		use_container_width=True,
	)
	st.caption(
		"The CSV contains your original dataset plus optimisation_decision: "
		"1 = selected future crop tree (crop-tree selection) or selected for removal "
		"(thinning); 0 = not selected."
	)


CSS = f"<style>{load_static_text(STATIC_DIR / 'dashboard.css')}</style>"


def initialize_state() -> None:
	if st.session_state.get("_case_study_mode") != 1:
		select_data_source(st.session_state.get("data_source"))
		st.session_state._case_study_mode = 1
	defaults = {
		"selected_problem": "future_crop_tree_selection",
		"selected_algorithm": "linear_programming",
		"epsg_text": "",
		"column_mapping": {
			"dbh": None,
			"species": None,
			"x_coord": None,
			"y_coord": None,
			"social_status": None,
			"is_alive": None,
			"wood_quality": None,
			"volume": None,
		},
		"objective_social_status": False,
		"objective_wood_quality": False,
		"objective_min_dbh": False,
		"objective_min_volume": False,
		"show_map_filter": False,
		"show_stand_boundaries_layer": True,
		"show_water_bodies_layer": True,
		"show_decision_map": False,
		"show_thinning_decision_map": False,
		"show_nearest_z_tree": False,
		"constraint_only_living_trees": True,
		"constraint_spatial_spread": False,
		"constraint_density": False,
		"constraint_density_min": 0,
		"constraint_density_max": 0,
		"constraint_min_distance": False,
		"constraint_min_distance_value": 5.0,
		"constraint_grid": False,
		"constraint_grid_type": "square",
		"constraint_grid_size": 10,
		"constraint_basal_area": False,
		"constraint_basal_area_keep_fraction": 0.5,
		"specify_species_distribution": False,
		"specify_dbh_distribution": False,
		"stand_shapefile_bytes": None,
		"stand_shapefile_name": None,
		"water_shapefile_bytes": None,
		"water_shapefile_name": None,
		"roads_gpkg_bytes": None,
		"roads_gpkg_name": None,
		"water_buffer_distance": 10.0,
		"road_buffer_distance": 3.0,
		"show_roads_layer": True,
		"stand_assignment_seed": 2026,
		"spatial_validation": None,
		"optimisation_stand_results": None,
		"optimisation_results": None,
		"optimisation_algorithm": None,
		"optimisation_dataset": None,
		"optimisation_objectives": None,
		"optimisation_constraints": None,
		"optimisation_benchmarks": None,
		"optimisation_optimiser": None,
		"optimisation_report_percentages": False,
		"diagnostic_stand_id": "",
		"n_thinning_treatments": 1,
		"thinning_use_basal_area": False,
		"thinning_density_targets": [],
		"thinning_basal_area_targets": [],
		"thinning_schedule": None,
		"desired_species_frequency": None,
		"desired_dbh_frequency": None,
		"desired_species_labels": [],
		"desired_dbh_edges": [],
		"report_objectives_as_percentages": False,
		"lp_solver": "CBC",
		"gurobi_wls_access_id": "",
		"gurobi_wls_secret": "",
		"gurobi_wls_license_id": "",
		"lp_max_runtime": 120,
		"de_population_size": 15,
		"de_iterations": 100,
		"da_iterations": 1000,
		"da_initial_temp": 5230.0,
		"da_restart_temp_ratio": 2e-5,
		"da_visit": 2.62,
		"da_accept": -5.0,
		"da_maxfun": 10000000,
		"ga_num_generations": 100,
		"ga_sol_per_pop": 50,
		"ga_num_parents_mating": 10,
		"ga_parent_selection_type": "sss",
		"ga_keep_elitism": 1,
		"ga_crossover_type": "single_point",
		"ga_mutation_type": "random",
		"ga_mutation_probability": 0.05,
		"ga_use_random_seed": False,
		"ga_random_seed": 0,
	}

	for key, value in defaults.items():
		st.session_state.setdefault(key, value)

	column_mapping_defaults = defaults["column_mapping"]
	for key, value in column_mapping_defaults.items():
		st.session_state.column_mapping.setdefault(key, value)
	# Hidden controls must not remain active in existing browser sessions.
	st.session_state.objective_min_dbh = False
	st.session_state.show_map_filter = False
	st.session_state.selected_algorithm = "linear_programming"
	preset = TEST_DATASETS.get(st.session_state.get("data_source"), {})
	st.session_state.selected_problem = preset.get("problem")
	st.session_state.epsg_text = preset.get("epsg", "")


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
	"""Parse uploaded CSV bytes once and reuse the result across reruns."""
	buffer = BytesIO(raw_bytes)
	try:
		return pd.read_csv(buffer)
	except Exception:
		buffer.seek(0)
		return pd.read_csv(buffer, sep=None, engine="python")


def safe_read_csv(uploaded_file) -> pd.DataFrame:
	return read_csv_bytes(uploaded_file.getvalue())


@st.cache_data(show_spinner=False)
def read_roads_geopackage(raw_bytes: bytes) -> gpd.GeoDataFrame:
	"""Read notebook skid trails or another line-only roads GeoPackage."""
	if not raw_bytes:
		raise ValueError("The uploaded roads GeoPackage is empty.")
	layers = gpd.list_layers(BytesIO(raw_bytes))
	layer_names = layers["name"].tolist()
	if "skid_roads" in layer_names:
		layer_name = "skid_roads"
	elif len(layer_names) == 1:
		layer_name = layer_names[0]
	else:
		raise ValueError("Use a single-layer roads GeoPackage or a layer named skid_roads.")
	roads = gpd.read_file(BytesIO(raw_bytes), layer=layer_name)
	if roads.crs is None:
		raise ValueError("The roads GeoPackage must specify its CRS.")
	if roads.empty or roads.geometry.isna().any() or roads.geometry.is_empty.any():
		raise ValueError("The roads GeoPackage must contain nonempty line geometries.")
	if not roads.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
		raise ValueError("Roads must be LineString or MultiLineString geometries.")
	if not roads.geometry.is_valid.all():
		raise ValueError("The roads GeoPackage contains invalid line geometries.")
	return roads


@st.cache_data(show_spinner=False)
def read_geometry_shapefile(raw_bytes: bytes, epsg_code: int) -> gpd.GeoDataFrame:
	"""Read polygon geometry directly from a standalone .shp upload."""
	if not raw_bytes:
		raise ValueError("The uploaded shapefile is empty.")
	reader = shapefile.Reader(shp=BytesIO(raw_bytes))
	shapes = [shape(item.__geo_interface__) for item in reader.iterShapes()]
	geometry = gpd.GeoDataFrame(geometry=shapes, crs=f"EPSG:{epsg_code}")
	if geometry.empty:
		raise ValueError("The shapefile contains no geometries.")
	geometry = geometry[["geometry"]].dropna(subset=["geometry"]).copy()
	geometry.geometry = geometry.geometry.make_valid()
	geometry = geometry.explode(index_parts=False, ignore_index=True)
	geometry = geometry[
		geometry.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
	].reset_index(drop=True)
	if geometry.empty:
		raise ValueError("The shapefile contains no polygon geometries.")
	return geometry


def _polygonal_components(geometry):
	"""Return only the polygonal area from a possibly mixed intersection."""
	if isinstance(geometry, (Polygon, MultiPolygon)):
		return geometry
	parts = []
	for component in getattr(geometry, "geoms", []):
		polygonal = _polygonal_components(component)
		if polygonal is not None and not polygonal.is_empty:
			parts.append(polygonal)
	return unary_union(parts) if parts else None


def validate_stand_boundaries(stands: gpd.GeoDataFrame) -> dict:
	"""Report the extent of overlaps between stand polygons."""
	overlap_pairs = []
	overlap_details = []
	spatial_index = stands.sindex
	for left_index, left_geometry in enumerate(stands.geometry):
		for right_index in spatial_index.query(left_geometry, predicate="intersects"):
			right_index = int(right_index)
			if right_index <= left_index:
				continue
			overlap = left_geometry.intersection(stands.geometry.iloc[right_index])
			polygonal_overlap = _polygonal_components(overlap)
			if polygonal_overlap is not None and polygonal_overlap.area > 1e-6:
				overlap_pairs.append((left_index, right_index))
				# Twice the radius of the largest contained circle gives the user a
				# useful linear measure of the overlap. Source polygons are not changed.
				max_width_m = 2.0 * maximum_inscribed_circle(
					polygonal_overlap, tolerance=0.01
				).length
				overlap_details.append({
					"stand_pair": (left_index, right_index),
					"area_m2": float(polygonal_overlap.area),
					"max_width_m": float(max_width_m),
				})
	return {
		"stand_count": len(stands),
		"overlap_pairs": overlap_pairs,
		"overlap_details": overlap_details,
	}


def stand_overlap_warning(validation: dict) -> str:
	"""Summarise stand overlaps without implying that polygons will be modified."""
	details = validation.get("overlap_details", [])
	total_area = sum(item["area_m2"] for item in details)
	max_width = max((item["max_width_m"] for item in details), default=0.0)
	return (
		f"Detected {len(details):,} overlapping stand pair(s), with {total_area:,.2f} m² "
		f"of summed pairwise overlap and an estimated maximum width of {max_width:,.2f} m. "
		"Stand boundaries will not be changed."
	)


def assign_trees_to_stands(
	dataset: SingleTreeDataset,
	stands: gpd.GeoDataFrame,
	random_seed: int,
) -> dict:
	"""Assign each tree to one stand, randomly resolving all multiple matches."""
	stands = stands.reset_index(drop=True).copy()
	stands["stand_id"] = [f"stand_{index + 1:03d}" for index in range(len(stands))]
	rng = np.random.default_rng(random_seed)
	spatial_index = stands.sindex
	assignments = []
	boundary_ties = 0
	for tree in dataset.data.geometry:
		matches = np.asarray(
			spatial_index.query(tree, predicate="intersects"), dtype=int
		)
		if matches.size == 0:
			assignments.append(None)
		elif matches.size == 1:
			assignments.append(stands.iloc[int(matches[0])]["stand_id"])
		else:
			boundary_ties += 1
			chosen = int(rng.choice(matches))
			assignments.append(stands.iloc[chosen]["stand_id"])
	dataset.data["CG_source_row"] = np.arange(dataset._n_trees, dtype=int)
	dataset.data["stand_id"] = assignments
	return {
		"stands": stands,
		"unassigned_trees": int(pd.isna(dataset.data["stand_id"]).sum()),
		"boundary_ties": boundary_ties,
	}


def spatial_buffer(geometry: gpd.GeoDataFrame, distance_metres: float) -> gpd.GeoDataFrame:
	"""Buffer in metres, including datasets using geographic or non-metre CRSs."""
	if not np.isfinite(distance_metres) or distance_metres < 0:
		raise ValueError("Buffer distance must be a finite, nonnegative number of metres.")
	metric_crs = geometry.crs
	if metric_crs is None:
		raise ValueError("Spatial data must specify a CRS.")
	if not metric_crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in metric_crs.axis_info[:2]):
		metric_crs = geometry.estimate_utm_crs()
	projected = geometry.to_crs(metric_crs)
	return gpd.GeoDataFrame(geometry=projected.geometry.buffer(float(distance_metres)), crs=metric_crs)


def apply_water_protection(dataset, water, distance_metres) -> int:
	"""Mark trees in water buffers; road clearing is applied afterwards."""
	mask = np.zeros(dataset._n_trees, dtype=bool)
	if water is not None and not water.empty:
		area = spatial_buffer(water, distance_metres).to_crs(dataset.data.crs)
		mask = dataset.data.geometry.intersects(unary_union(area.geometry)).to_numpy(dtype=bool)
	dataset.data[WaterProtectionConstraint.indicator_col] = mask
	return int(mask.sum())


def apply_road_harvest(dataset, roads, distance_metres) -> int:
	"""Force road clearing, overriding water protection where buffers overlap."""
	mask = np.zeros(dataset._n_trees, dtype=bool)
	if roads is not None and not roads.empty:
		area = spatial_buffer(roads, distance_metres) if distance_metres > 0 else roads
		area = area.to_crs(dataset.data.crs)
		mask = dataset.data.geometry.intersects(unary_union(area.geometry)).to_numpy(dtype=bool)
	dataset.data[RoadHarvestConstraint.indicator_col] = mask
	if WaterProtectionConstraint.indicator_col in dataset.data:
		dataset.data.loc[mask, WaterProtectionConstraint.indicator_col] = False
	return int(mask.sum())


def buffer_eligible_mask(data: pd.DataFrame) -> np.ndarray:
	"""Identify trees whose harvest/retention is not predetermined by buffers."""
	eligible = np.ones(len(data), dtype=bool)
	for column in (WaterProtectionConstraint.indicator_col, RoadHarvestConstraint.indicator_col):
		if column in data:
			eligible &= ~data[column].fillna(False).to_numpy(dtype=bool)
	return eligible


def prepare_buffer_statistics(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
	"""Use the same buffer masks for pre-optimisation statistics and the solver."""
	if df is None or not is_thinning_treatment():
		return df
	water_bytes = st.session_state.get("water_shapefile_bytes")
	road_bytes = st.session_state.get("roads_gpkg_bytes")
	if not (water_bytes or road_bytes):
		return df
	try:
		dataset = build_dataset(df, st.session_state.column_mapping, st.session_state.epsg_text)
		water = read_geometry_shapefile(water_bytes, dataset.epsg) if water_bytes else None
		roads = read_roads_geopackage(road_bytes) if road_bytes else None
		apply_water_protection(dataset, water, st.session_state.water_buffer_distance)
		road_count = apply_road_harvest(dataset, roads, st.session_state.road_buffer_distance)
		eligible = buffer_eligible_mask(dataset.data)
		st.caption(f"Distributions cover {int(eligible.sum()):,} eligible trees; exclude {road_count:,} road-clearing trees and {int(dataset.data[WaterProtectionConstraint.indicator_col].sum()):,} water-protected trees. Road clearing takes priority in overlaps.")
		return dataset.data.loc[eligible].copy()
	except Exception as exc:
		st.info(f"Buffer-adjusted distributions need valid coordinate mappings, EPSG code and spatial inputs: {exc}")
		return None


def validate_buffer_decisions(dataset, result) -> None:
	vector = np.asarray(result["decision_vector"], dtype=bool)
	for column, expected in ((RoadHarvestConstraint.indicator_col, True), (WaterProtectionConstraint.indicator_col, False)):
		if column in dataset.data:
			mask = dataset.data[column].to_numpy(dtype=bool)
			if np.any(vector[mask] != expected):
				raise ValueError("The solver did not satisfy mandatory road clearing / water protection. No valid buffer-compliant result was found; try linear programming or revise the other constraints.")


def render_roads_layer_control(dataset) -> Optional[gpd.GeoDataFrame]:
	road_bytes = getattr(dataset, "buffer_inputs", st.session_state).get("roads_gpkg_bytes")
	st.checkbox("Roads and harvest buffer", key="show_roads_layer", disabled=not bool(road_bytes))
	return read_roads_geopackage(road_bytes) if road_bytes and st.session_state.show_roads_layer else None


def set_state_value(key: str, value) -> None:
	st.session_state[key] = value


def select_problem(problem_key: str) -> None:
	"""Select a treatment and clear controls that do not transfer between modes."""
	if problem_key != TEST_DATASETS.get(st.session_state.get("data_source"), {}).get("problem"):
		return
	if st.session_state.get("selected_problem") != problem_key:
		for key in (
			"objective_social_status", "objective_wood_quality", "objective_min_dbh",
			"objective_min_volume",
			"constraint_only_living_trees", "specify_species_distribution", "specify_dbh_distribution",
			"constraint_spatial_spread", "constraint_density",
			"constraint_min_distance", "constraint_grid",
			"constraint_basal_area",
		):
			st.session_state[key] = False
	st.session_state.selected_problem = problem_key


def is_thinning_treatment() -> bool:
	return st.session_state.get("selected_problem") in {
		"thinning_treatment", "risk_mitigation",
	}


def is_thinning_from_above() -> bool:
	return st.session_state.get("selected_problem") == "risk_mitigation"


def objective_scores_from_results(objectives: List[object], decisions: List[int]) -> List[Dict[str, float]]:
	scores = []
	for objective in objectives:
		decision_vector = np.zeros(objective.dataset._n_trees, dtype=np.int8)
		decision_vector[decisions] = 1
		try:
			score = objective.calculate(decision_vector)
		except TypeError:
			score = objective.calculate(objective.dataset, decision_vector)
		scores.append({"objective": objective.name, "score": float(score)})
	return scores


def render_objective_diagnostics(objectives: List[object], decisions: List[int]) -> None:
	if not objectives:
		return

	scores = objective_scores_from_results(objectives, decisions)
	render_objective_score_cards(scores)


def prepare_map_data(
	df: pd.DataFrame,
	x_col: str,
	y_col: str,
	epsg_code: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
	"""Prepare valid source-CRS and WGS84 point data for map rendering."""
	coords = df[[x_col, y_col]].copy()
	coords[x_col] = pd.to_numeric(coords[x_col], errors="coerce")
	coords[y_col] = pd.to_numeric(coords[y_col], errors="coerce")
	coords = coords.dropna(subset=[x_col, y_col])
	if coords.empty:
		return None

	gdf = gpd.GeoDataFrame(
		coords,
		geometry=gpd.points_from_xy(coords[x_col], coords[y_col]),
		crs=f"EPSG:{epsg_code}",
	)
	return gdf, gdf.to_crs(epsg=4326)


def _legacy_folium_decision_map(
	df: pd.DataFrame,
	x_col: str,
	y_col: str,
	epsg_code: int,
	decisions: List[int],
	constraints: Optional[List[object]] = None,
	grid_squares: Optional[gpd.GeoDataFrame] = None,
	treatment_type: str = "future_crop_tree_selection",
) -> None:
	map_data = prepare_map_data(df, x_col, y_col, epsg_code)
	if map_data is None:
		st.warning("No valid coordinate rows were found for the decision map.")
		return

	gdf, gdf_wgs84 = map_data
	selected_indices = set(decisions)
	selected_row_labels = set(df.index.take(decisions))
	if treatment_type == "thinning_treatment":
		selected_colour, other_colour = "#D55E00", "#009E73"
		selected_label, other_label = "Selected for removal", "Left standing"
	else:
		selected_colour, other_colour = "#0072B2", "#7f8c8d"
		selected_label, other_label = "Future crop trees", "Other trees"
	center_lat = float(gdf_wgs84.geometry.y.mean())
	center_lon = float(gdf_wgs84.geometry.x.mean())
	m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

	if grid_squares is not None and not grid_squares.empty:
		folium.GeoJson(
			grid_squares.to_crs(epsg=4326),
			name="Grid boundaries",
			style_function=lambda _: {
				"fillColor": "transparent",
				"color": "#4a5568",
				"weight": 1,
				"fillOpacity": 0,
				"opacity": 0.55,
			},
		).add_to(m)

	min_distance_constraint = next(
		(
			constraint
			for constraint in constraints or []
			if getattr(constraint, "type", None) == "pairwise_distance"
		),
		None,
	)
	min_distance = getattr(min_distance_constraint, "min_distance", None)
	if min_distance is not None:
		for index, row in gdf_wgs84.iterrows():
			if index in selected_row_labels:
				folium.Circle(
					location=[row.geometry.y, row.geometry.x],
					radius=float(min_distance),
					color="#d35400",
					fill=True,
					fill_color="#f39c12",
					fill_opacity=0.12,
					opacity=0.65,
					weight=1,
				).add_to(m)

	for index, row in gdf_wgs84.iterrows():
		selected = index in selected_row_labels
		color = selected_colour if selected else other_colour
		data_row = df.loc[index]
		tooltip_html = "<br>".join(
			f"{html.escape(str(column))}: {html.escape(str(data_row[column]))}"
			for column in df.columns[:10]
		)
		folium.CircleMarker(
			location=[row.geometry.y, row.geometry.x],
			radius=4,
			color=color,
			fill=True,
			fill_color=color,
			fill_opacity=0.9,
			weight=1,
			tooltip=folium.Tooltip(tooltip_html),
		).add_to(m)

	distance_legend_html = ""
	if min_distance is not None:
		distance_legend_html = f"""
		<div style="display: flex; align-items: center; margin-top: 4px;">
			<span style="height: 14px; width: 14px; border: 2px solid #d35400; background-color: rgba(243,156,18,0.2); border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
			Minimum distance radius ({float(min_distance):g} m)
		</div>
		"""
	grid_legend_html = ""
	if grid_squares is not None and not grid_squares.empty:
		grid_legend_html = """
		<div style="display: flex; align-items: center; margin-top: 4px;">
			<span style="height: 12px; width: 18px; border: 2px solid #4a5568; display: inline-block; margin-right: 8px;"></span>
			Grid boundary
		</div>
		"""

	legend_html = f"""
	<div style="
		position: fixed;
		bottom: 30px;
		left: 30px;
		z-index: 9999;
		background-color: white;
		padding: 10px 14px;
		border: 2px solid #444;
		border-radius: 6px;
		font-size: 14px;
		box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
	">
		<div style="margin-bottom: 4px;"><b>Legend</b></div>
		<div style="display: flex; align-items: center; margin-bottom: 4px;">
			<span style="height: 12px; width: 12px; background-color: {selected_colour}; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
			{html.escape(selected_label)}: {len(selected_indices):,} trees
		</div>
		<div style="display: flex; align-items: center;">
			<span style="height: 12px; width: 12px; background-color: {other_colour}; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
			{html.escape(other_label)}: {len(df) - len(selected_indices):,} trees
		</div>
		{grid_legend_html}
		{distance_legend_html}
	</div>
	"""
	m.get_root().html.add_child(folium.Element(legend_html))

	if st_folium is not None:
		st_folium(m, width=None, height=420)
	elif components is not None:
		components.html(m._repr_html_(), height=440, scrolling=False)
	else:
		st.warning("Folium map preview is unavailable in this environment.")


def render_pydeck_decision_map(
	df: pd.DataFrame,
	x_col: str,
	y_col: str,
	epsg_code: int,
	decisions: List[int],
	id_col: str,
	dbh_col: Optional[str] = None,
	species_col: Optional[str] = None,
	treatment_type: str = "future_crop_tree_selection",
	decision_groups: Optional[Dict[str, List[int]]] = None,
	competition_indices: Optional[np.ndarray] = None,
	nearest_z_tree_indices: Optional[np.ndarray] = None,
	show_nearest_z_tree: bool = False,
	stand_boundaries: Optional[gpd.GeoDataFrame] = None,
	water_bodies: Optional[gpd.GeoDataFrame] = None,
	water_buffer_distance: float = 0.0,
	roads: Optional[gpd.GeoDataFrame] = None,
	road_buffer_distance: float = 0.0,
	selected_tree_circle_radius: float = 0.0,
) -> None:
	"""Render a lightweight, non-navigable WebGL decision map."""
	map_data = prepare_map_data(df, x_col, y_col, epsg_code)
	if map_data is None:
		st.warning("No valid coordinate rows were found for the decision map.")
		return

	gdf_source, gdf_wgs84 = map_data
	selected_row_labels = set(df.index.take(decisions))
	row_positions = df.index.get_indexer(gdf_wgs84.index)
	map_df = pd.DataFrame(
		{
			"longitude": gdf_wgs84.geometry.x.to_numpy(),
			"latitude": gdf_wgs84.geometry.y.to_numpy(),
			"tree_position": row_positions,
		},
		index=gdf_wgs84.index,
	)
	map_df["tree_id"] = df.loc[map_df.index, id_col].astype(str).to_numpy()
	map_df["dbh"] = (
		df.loc[map_df.index, dbh_col].astype(str).to_numpy()
		if dbh_col is not None and dbh_col in df.columns
		else "N/A"
	)
	map_df["species"] = (
		df.loc[map_df.index, species_col].astype(str).to_numpy()
		if species_col is not None and species_col in df.columns
		else "N/A"
	)
	map_df["stand_id"] = (
		df.loc[map_df.index, "stand_id"].fillna("N/A").astype(str).to_numpy()
		if "stand_id" in df.columns else "N/A"
	)
	if competition_indices is not None and nearest_z_tree_indices is not None:
		map_df["competition_index"] = competition_indices[row_positions]
		map_df["competition_index_display"] = map_df["competition_index"].map(
			lambda value: "N/A" if pd.isna(value) else f"{value:.5f}"
		)
		map_df["nearest_z_tree_id"] = df.iloc[
			nearest_z_tree_indices[row_positions]
		][id_col].astype(str).to_numpy()
		map_df.loc[map_df.index.isin(set(df.index.take(decisions))), [
			"competition_index", "nearest_z_tree_id"
		]] = [np.nan, "N/A"]

	if decision_groups:
		group_colours = {
			"Future crop trees": [213, 94, 0, 235],
			"Thinning 1": [0, 90, 70, 235],
			"Thinning 2": [0, 117, 94, 235],
			"Thinning 3": [0, 158, 115, 235],
			"Thinning 4": [72, 160, 100, 235],
			"Thinning 5": [126, 183, 110, 235],
			"Left standing": [127, 140, 140, 220],
		}
		map_df["decision"] = "Left standing"
		layers = []
		for group_name, indices in decision_groups.items():
			group_mask = map_df.index.isin(set(df.index.take(indices)))
			map_df.loc[group_mask, "decision"] = group_name
			layers.append(pdk.Layer(
				"ScatterplotLayer",
				data=map_df.loc[group_mask],
				id=f"trees-{group_name.lower().replace(' ', '-')}",
				get_position=["longitude", "latitude"],
				get_fill_color=group_colours.get(group_name, [80, 80, 80, 235]),
				get_radius=4,
				radius_units="pixels",
				pickable=True,
			))
		left_mask = map_df["decision"] == "Left standing"
		layers.append(pdk.Layer(
			"ScatterplotLayer", data=map_df.loc[left_mask], id="trees-left-standing",
			get_position=["longitude", "latitude"], get_fill_color=group_colours["Left standing"],
			get_radius=3, radius_units="pixels", pickable=True,
		))
		legend_entries = [
			(group_name, group_colours.get(group_name, [80, 80, 80, 235]))
			for group_name in decision_groups
		]
		legend_entries.append(("Left standing", group_colours["Left standing"]))
	else:
		selected_mask = map_df.index.isin(selected_row_labels)
		if treatment_type == "thinning_treatment":
			selected_decision = "Selected for removal"
			other_decision = "Left standing"
		else:
			selected_decision = "Future crop tree"
			other_decision = "Other tree"
		map_df["decision"] = np.where(selected_mask, selected_decision, other_decision)
		layers = [
			pdk.Layer("ScatterplotLayer", data=map_df.loc[~selected_mask], id="trees-not-selected",
				get_position=["longitude", "latitude"], get_fill_color=[0, 158, 115, 220],
				get_radius=3, radius_units="pixels", pickable=True),
			pdk.Layer("ScatterplotLayer", data=map_df.loc[selected_mask], id="trees-selected",
				get_position=["longitude", "latitude"], get_fill_color=[213, 94, 0, 235],
				get_radius=4, radius_units="pixels", pickable=True),
		]
		legend_entries = [
			(other_decision, [0, 158, 115, 220]),
			(selected_decision, [213, 94, 0, 235]),
		]

	if show_nearest_z_tree:
		if competition_indices is None or nearest_z_tree_indices is None:
			raise ValueError("Nearest-Z map data is required to show nearest Z trees.")
		z_tree_indices = set(df.index.take(decisions))
		coordinates_by_position = {
			int(position): geometry
			for position, geometry in zip(
				row_positions, gdf_wgs84.geometry, strict=True
			)
		}
		line_geometries = []
		for source_index, source_position in zip(
			gdf_wgs84.index, row_positions, strict=True
		):
			if source_index in z_tree_indices:
				continue
			target_position = int(nearest_z_tree_indices[source_position])
			source_point = coordinates_by_position[int(source_position)]
			target_point = coordinates_by_position[target_position]
			line_geometries.append(LineString([
				(source_point.x, source_point.y),
				(target_point.x, target_point.y),
			]))
		line_data = gpd.GeoDataFrame(
			{"geometry": line_geometries}, crs="EPSG:4326"
		)
		layers.insert(0, pdk.Layer(
			"GeoJsonLayer",
			data=json.loads(line_data.to_json()),
			id="nearest-z-tree-lines",
			get_line_color=[80, 80, 80, 150],
			get_line_width=1,
			line_width_units="pixels",
		))

	if selected_tree_circle_radius > 0 and selected_row_labels:
		selected_points = gdf_source.loc[gdf_source.index.isin(selected_row_labels)].copy()
		metric_crs = (
			selected_points.estimate_utm_crs()
			if selected_points.crs is not None and selected_points.crs.is_geographic
			else selected_points.crs
		)
		if metric_crs is None:
			raise ValueError("A valid CRS is required to draw metric distance circles.")
		distance_circles = gpd.GeoDataFrame(
			geometry=selected_points.to_crs(metric_crs).geometry.buffer(
				float(selected_tree_circle_radius)
			),
			crs=metric_crs,
		).to_crs(4326)
		layers.insert(0, pdk.Layer(
			"GeoJsonLayer",
			data=json.loads(distance_circles.to_json()),
			id="selected-tree-distance-circles",
			filled=True,
			stroked=True,
			get_fill_color=[213, 94, 0, 30],
			get_line_color=[213, 94, 0, 180],
			get_line_width=1,
			line_width_units="pixels",
			pickable=False,
		))
		legend_entries.append((
			f"Selected-tree spacing radius ({selected_tree_circle_radius:g} m)",
			[213, 94, 0, 80],
		))

	overlay_layers = []
	if water_bodies is not None and not water_bodies.empty:
		water_projected = water_bodies.to_crs(epsg_code)
		if water_buffer_distance > 0:
			water_buffer = spatial_buffer(water_projected, water_buffer_distance).to_crs(4326)
			overlay_layers.append(pdk.Layer(
				"GeoJsonLayer", data=json.loads(water_buffer.to_json()), id="water-buffer",
				filled=True, stroked=True, get_fill_color=[86, 180, 233, 65],
				get_line_color=[0, 114, 178, 180], get_line_width=1,
				line_width_units="pixels", pickable=False,
			))
		water_wgs84 = water_projected.to_crs(4326)
		overlay_layers.append(pdk.Layer(
			"GeoJsonLayer", data=json.loads(water_wgs84.to_json()), id="water-bodies",
			filled=True, stroked=True, get_fill_color=[0, 114, 178, 150],
			get_line_color=[0, 70, 140, 230], get_line_width=2,
			line_width_units="pixels", pickable=False,
		))
		legend_entries.append(("Water bodies", [0, 114, 178, 255]))
		if water_buffer_distance > 0:
			legend_entries.append((f"Water buffer ({water_buffer_distance:g} m)", [86, 180, 233, 255]))

	if roads is not None and not roads.empty:
		if road_buffer_distance > 0:
			road_buffer = spatial_buffer(roads, road_buffer_distance).to_crs(4326)
			overlay_layers.append(pdk.Layer(
				"GeoJsonLayer", data=json.loads(road_buffer.to_json()), id="road-buffer",
				filled=True, stroked=True, get_fill_color=[230, 159, 0, 65],
				get_line_color=[230, 159, 0, 180], get_line_width=1,
				line_width_units="pixels", pickable=False,
			))
			legend_entries.append((f"Road harvest buffer ({road_buffer_distance:g} m)", [230, 159, 0, 255]))
		overlay_layers.append(pdk.Layer(
			"GeoJsonLayer", data=json.loads(roads.to_crs(4326).to_json()), id="roads",
			filled=False, stroked=True, get_line_color=[139, 69, 19, 255],
			get_line_width=3, line_width_units="pixels", pickable=False,
		))
		legend_entries.append(("Roads / skid trails", [139, 69, 19, 255]))

	if stand_boundaries is not None and not stand_boundaries.empty:
		stands_wgs84 = stand_boundaries.to_crs(4326)
		overlay_layers.append(pdk.Layer(
			"GeoJsonLayer", data=json.loads(stands_wgs84.to_json()), id="stand-boundaries",
			filled=False, stroked=True, get_line_color=[20, 20, 20, 230],
			get_line_width=2, line_width_units="pixels", pickable=False,
		))
		legend_entries.append(("Stand boundaries", [20, 20, 20, 255]))

	layers = overlay_layers + layers

	center_lat = float(map_df["latitude"].mean())
	center_lon = float(map_df["longitude"].mean())
	longitude_span = float(map_df["longitude"].max() - map_df["longitude"].min())
	latitude_span = float(map_df["latitude"].max() - map_df["latitude"].min())
	span = max(longitude_span * max(np.cos(np.radians(center_lat)), 0.01), latitude_span)
	zoom = 20.0 if span <= 0 else float(np.clip(np.log2(360.0 / span) - 1.0, 1.0, 20.0))

	tooltip_text = "ID: {tree_id}\nStand: {stand_id}\nDBH: {dbh}\nSpecies: {species}\nDecision: {decision}"
	if competition_indices is not None and nearest_z_tree_indices is not None:
		tooltip_text += "\nCompetition index: {competition_index_display}\nNearest Z tree: {nearest_z_tree_id}"

	legend_items = "".join(
		f'<span style="display:inline-flex;align-items:center;margin:0 16px 6px 0;">'
		f'<span style="width:12px;height:12px;border-radius:2px;margin-right:6px;'
		f'background:rgba({colour[0]},{colour[1]},{colour[2]},{colour[3] / 255:.2f});'
		f'border:1px solid #555;"></span>{html.escape(label)}</span>'
		for label, colour in legend_entries
	)
	st.markdown(f"<div><strong>Legend</strong><br>{legend_items}</div>", unsafe_allow_html=True)

	deck = pdk.Deck(
		layers=layers,
		map_style=None,
		initial_view_state=pdk.ViewState(
			latitude=center_lat,
			longitude=center_lon,
			zoom=zoom,
			pitch=0,
			bearing=0,
			controller=False,
		),
		tooltip={"text": tooltip_text},
	)
	st.pydeck_chart(deck, use_container_width=True)

@st.cache_data(show_spinner=False)
def build_dataset(df: pd.DataFrame, mapping: Dict[str, Optional[str]], epsg_text: str) -> SingleTreeDataset:
	"""Build and spatially prepare a dataset once for each input configuration."""
	dataset = SingleTreeDataset(df.copy())
	dataset.source_data = df.copy()
	col_mapping = {}
	for source_key, target_key in (
		("id", "id"),
		("dbh", "dbh"),
		("x_coord", "x"),
		("y_coord", "y"),
		("species", "species"),
		("social_status", "social_status"),
		("is_alive", "is_alive"),
		("wood_quality", "quality"),
		("volume", "volume"),
	):
		value = mapping.get(source_key)
		if isinstance(value, str) and value:
			col_mapping[target_key] = value
	dataset.set_columns(**col_mapping)
	dataset.convert_mapped_column_types()
	# The EPSG setter already geodatafies the data and calculates its area.
	dataset.epsg = int(epsg_text)

	return dataset


def build_objectives(dataset: SingleTreeDataset, obj_weights: dict[str, float]) -> Dict[object, float]:
	objectives = {}
	if st.session_state.objective_min_volume:
		volume_objective = MaxVolumeObjective() if is_thinning_from_above() else MinVolumeObjective()
		objectives[volume_objective] = obj_weights.get("minvolume", 1.0)
	if st.session_state.objective_social_status:
		objectives[SocialStatusObjective()] = obj_weights.get("socialstatus", None)
	if st.session_state.objective_wood_quality:
		objectives[QualityObjective()] = obj_weights.get("woodquality", None)
	if st.session_state.objective_min_dbh:
		objectives[MinDBHObjective()] = obj_weights.get("mindbh", None)
	return objectives


def total_basal_area_per_hectare(dataset: SingleTreeDataset, eligible_only: bool = False) -> float:
	"""Calculate stand basal area from centimetre DBH values."""
	dbh_col = dataset._get_column_name("dbh")
	diameters_m = dataset.data[dbh_col].to_numpy(dtype=float) / 100
	if eligible_only:
		diameters_m = diameters_m[buffer_eligible_mask(dataset.data)]
	return float(np.sum(np.pi * (diameters_m / 2) ** 2) / dataset.hectares)


def tree_basal_areas_per_hectare(dataset: SingleTreeDataset) -> np.ndarray:
	"""Return each tree's basal-area contribution in m2/ha."""
	dbh_col = dataset._get_column_name("dbh")
	diameters_m = dataset.data[dbh_col].to_numpy(dtype=float) / 100
	return np.pi * (diameters_m / 2) ** 2 / dataset.hectares


def stand_basal_area_metrics(
	dataset: SingleTreeDataset,
	decision_vector: Optional[np.ndarray] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
	"""Return original BA, retained BA, and retained percentage for one stand."""
	if not dataset.columns.get("dbh") or dataset.hectares <= 0:
		return None, None, None
	contributions = tree_basal_areas_per_hectare(dataset)
	original = float(contributions.sum())
	if decision_vector is None:
		return original, None, None
	cut_mask = np.asarray(decision_vector, dtype=bool)
	if cut_mask.shape != contributions.shape:
		raise ValueError("Decision vector does not match the stand tree count.")
	retained = float(contributions[~cut_mask].sum())
	percentage = 100.0 * retained / original if original > 0 else None
	return original, retained, percentage


def render_frequency_specifications(df: Optional[pd.DataFrame]) -> None:
	"""Render optional editable species and DBH frequency distributions."""
	if df is None:
		return
	configuration_columns = [column for column in (st.session_state.column_mapping.get("id"), st.session_state.column_mapping.get("species"), st.session_state.column_mapping.get("dbh")) if column in df.columns]
	configuration = hashlib.sha256(pd.util.hash_pandas_object(df[configuration_columns], index=True).values.tobytes()).hexdigest()[:16]
	if st.session_state.get("frequency_inventory_configuration") != configuration:
		st.session_state.desired_species_frequency = None
		st.session_state.desired_dbh_frequency = None
		st.session_state.desired_species_labels = []
		st.session_state.desired_dbh_edges = []
		st.session_state.frequency_inventory_configuration = configuration
	if st.session_state.specify_species_distribution:
		species_column = st.session_state.column_mapping.get("species")
		if isinstance(species_column, str) and species_column in df.columns:
			values = df[species_column].astype("string")
			labels = sorted(values.dropna().unique().tolist())
			actual = [int((values == label).sum()) for label in labels]
			previous = st.session_state.get("desired_species_frequency")
			if not isinstance(previous, dict):
				previous = {}
			with st.container(border=True):
				st.session_state.desired_species_frequency = frequency_histogram(
					labels, actual,
					previous.get("desired_counts", actual),
					previous.get("tolerances"),
					"Species", key=f"desired-species-frequency-{configuration}",
				)
			st.session_state.desired_species_labels = labels
	if st.session_state.specify_dbh_distribution:
		dbh_column = st.session_state.column_mapping.get("dbh")
		if isinstance(dbh_column, str) and dbh_column in df.columns:
			values = pd.to_numeric(df[dbh_column], errors="coerce").to_numpy(dtype=float)
			finite = np.isfinite(values) & (values >= 0)
			if finite.any():
				edges = np.arange(0.0, 5.0 * np.ceil(values[finite].max() / 5.0) + 5.0, 5.0)
				actual, _ = np.histogram(values[finite], bins=edges)
				labels = [f"{int(left)}-{int(right)}" for left, right in zip(edges[:-1], edges[1:])]
				previous = st.session_state.get("desired_dbh_frequency")
				if not isinstance(previous, dict):
					previous = {}
				with st.container(border=True):
					st.session_state.desired_dbh_frequency = frequency_histogram(
						labels, actual.tolist(),
						previous.get("desired_counts", actual.tolist()),
						previous.get("tolerances"),
						"DBH (cm)", key=f"desired-dbh-frequency-{configuration}",
					)
				st.session_state.desired_dbh_edges = edges.tolist()


def build_frequency_constraints(dataset: SingleTreeDataset) -> list[FrequencyConstraint]:
	"""Build count constraints for the enabled retained-tree distributions."""
	constraints = []
	eligible = buffer_eligible_mask(dataset.data)
	decision_scope = DecisionScope.SELECT if not is_thinning_treatment() else DecisionScope.NOT_SELECT
	if st.session_state.get("specify_species_distribution") and st.session_state.get("desired_species_frequency"):
		column = dataset._get_column_name("species")
		values = dataset.data[column].astype("string")
		labels = st.session_state.get("desired_species_labels", [])
		settings = st.session_state.desired_species_frequency
		masks = [(values == label).fillna(False).to_numpy(dtype=bool) & eligible for label in labels]
		bounds = []
		for desired, tolerance, actual in zip(settings["desired_counts"], settings["tolerances"], [int(mask.sum()) for mask in masks]):
			bounds.append((max(0, int(desired) - int(tolerance)), min(actual, int(desired) + int(tolerance))))
		constraints.append(FrequencyConstraint("Species frequency", masks, bounds, decision_scope))
	if st.session_state.get("specify_dbh_distribution") and st.session_state.get("desired_dbh_frequency"):
		column = dataset._get_column_name("dbh")
		values = pd.to_numeric(dataset.data[column], errors="coerce").to_numpy(dtype=float)
		edges = np.asarray(st.session_state.get("desired_dbh_edges", []), dtype=float)
		settings = st.session_state.desired_dbh_frequency
		masks = [(values >= left) & (values < right) & eligible for left, right in zip(edges[:-1], edges[1:])]
		bounds = []
		for desired, tolerance, mask in zip(settings["desired_counts"], settings["tolerances"], masks):
			actual = int(mask.sum())
			bounds.append((max(0, int(desired) - int(tolerance)), min(actual, int(desired) + int(tolerance))))
		constraints.append(FrequencyConstraint("DBH frequency", masks, bounds, decision_scope))
	return constraints


def build_constraints(dataset: SingleTreeDataset) -> List[object]:
	constraints = []
	frequency_constraints = build_frequency_constraints(dataset)
	if is_thinning_treatment():
		constraints.extend(frequency_constraints)
		if st.session_state.constraint_min_distance:
			constraints.append(PairwiseDistanceConstraint(
				min_distance=st.session_state.constraint_min_distance_value,
				applies_to=DecisionScope.NOT_SELECT,
				exclude_col=(
					WaterProtectionConstraint.indicator_col
					if st.session_state.get("water_shapefile_bytes") else None
				),
			))
		if st.session_state.constraint_basal_area:
			total_basal_area = total_basal_area_per_hectare(dataset, eligible_only=True)
			dataset.data["CG_buffer_eligible"] = buffer_eligible_mask(dataset.data)
			keep_fraction = float(np.clip(
				st.session_state.constraint_basal_area_keep_fraction, 0.0, 1.0
			))
			constraints.append(BasalAreaConstraint(
				basal_area_target=keep_fraction * total_basal_area,
				margin=0.05 * total_basal_area,
				eligible_col="CG_buffer_eligible",
			))
		if st.session_state.get("water_shapefile_bytes"):
			constraints.append(WaterProtectionConstraint())
		if st.session_state.get("roads_gpkg_bytes"):
			constraints.append(RoadHarvestConstraint())
		return constraints
	if st.session_state.constraint_only_living_trees:
		constraints.append(DeadTreeConstraint())
	constraints.extend(frequency_constraints)
	if st.session_state.constraint_density:
		density_min = st.session_state.constraint_density_min
		density_max = st.session_state.constraint_density_max
		constraints.append(ZTreeProportionConstraint(lb_trees_per_ha=density_min, ub_trees_per_ha=density_max))
	if st.session_state.constraint_spatial_spread:
		constraints.append(ZTreeSpatialConstraint(lb_trees_per_ha=st.session_state.constraint_density_min, ub_trees_per_ha=st.session_state.constraint_density_max))
	if st.session_state.constraint_min_distance:
		constraints.append(PairwiseDistanceConstraint(min_distance=st.session_state.constraint_min_distance_value))
	if st.session_state.constraint_grid:
		constraints.append(SpatialGridConstraint(
			grid_type=st.session_state.constraint_grid_type,
			grid_size=st.session_state.constraint_grid_size,
		))
	return constraints

def build_optimiser(
	dataset: SingleTreeDataset,
	objectives: Dict[object, float],
	constraints: List[object],
) -> TreeOptimiser:
	algorithm_config = ALGORITHM_METADATA.get(st.session_state.selected_algorithm)
	algorithm_class = algorithm_config["class"] if algorithm_config else None
	if algorithm_class is None:
		raise ValueError(f"Unsupported algorithm: {st.session_state.selected_algorithm}")

	optimiser_class = ThinningOptimiser if is_thinning_treatment() else FutureCropTreeOptimiser
	return optimiser_class(
		dataset=dataset,
		objectives=objectives,
		constraints=constraints,
		algorithm=algorithm_class(),
	)


def stand_dataset_from_global(
	dataset: SingleTreeDataset,
	stand_id: str,
	stand_geometry,
) -> SingleTreeDataset:
	"""Create an optimiser dataset whose area is the management stand polygon."""
	stand_data = dataset.data.loc[dataset.data["stand_id"].eq(stand_id)].copy()
	stand_data = stand_data.reset_index(drop=True)
	stand_dataset = SingleTreeDataset(stand_data)
	stand_dataset.columns = dataset.columns.copy()
	stand_dataset._epsg = dataset.epsg
	stand_dataset.hectares = float(stand_geometry.area) / 10000.0
	return stand_dataset


def clear_gurobi_credentials() -> None:
	for key in ("gurobi_wls_access_id", "gurobi_wls_secret", "gurobi_wls_license_id"):
		st.session_state[key] = ""


def uses_gurobi_wls() -> bool:
	return (
		st.session_state.selected_algorithm == "linear_programming"
		and st.session_state.lp_solver == "GUROBI"
	)


def session_wls_credentials() -> dict:
	return wls_credentials(
		st.session_state.get("gurobi_wls_access_id", ""),
		st.session_state.get("gurobi_wls_secret", ""),
		st.session_state.get("gurobi_wls_license_id", ""),
	)


def run_with_solver_license(operation, *args, **kwargs):
	"""Keep WLS credentials out of optimisers, results and Streamlit caches."""
	if uses_gurobi_wls():
		with gurobi_wls_session(session_wls_credentials()):
			return operation(*args, **kwargs)
	return operation(*args, **kwargs)


def lp_hyperparameters() -> dict:
	return {
		"solver_name": "GUROBI" if uses_gurobi_wls() else "CBC",
		"max_time": float(st.session_state.lp_max_runtime),
		"num_workers": 2,
	}


def batch_algorithm_hyperparameters() -> dict:
	"""Return non-interactive algorithm settings for one stand-sized solve."""
	algorithm = st.session_state.selected_algorithm
	if algorithm == "linear_programming":
		return lp_hyperparameters()
	if algorithm == "differential_evolution":
		return {
			"maxiter": int(st.session_state.de_iterations),
			"popsize": int(st.session_state.de_population_size),
		}
	if algorithm == "dual_annealing":
		return {
			"maxiter": int(st.session_state.da_iterations),
			"initial_temp": float(st.session_state.da_initial_temp),
			"restart_temp_ratio": float(st.session_state.da_restart_temp_ratio),
			"visit": float(st.session_state.da_visit),
			"accept": float(st.session_state.da_accept),
			"maxfun": int(st.session_state.da_maxfun),
		}
	if algorithm == "genetic_algorithm":
		return {
			"num_generations": int(st.session_state.ga_num_generations),
			"sol_per_pop": int(st.session_state.ga_sol_per_pop),
			"num_parents_mating": int(st.session_state.ga_num_parents_mating),
			"parent_selection_type": st.session_state.ga_parent_selection_type,
			"keep_elitism": int(st.session_state.ga_keep_elitism),
			"crossover_type": st.session_state.ga_crossover_type,
			"mutation_type": st.session_state.ga_mutation_type,
			"mutation_probability": float(st.session_state.ga_mutation_probability),
			"random_seed": (
				int(st.session_state.ga_random_seed)
				if st.session_state.ga_use_random_seed else None
			),
		}
	raise ValueError(f"Unsupported algorithm: {algorithm}")


def optimise_stands(
	dataset: SingleTreeDataset,
	stands: gpd.GeoDataFrame,
	obj_weights: dict[str, float],
) -> tuple[dict, list[dict]]:
	"""Solve one independent thinning problem per stand and recombine decisions."""
	combined_vector = dataset.data.get(RoadHarvestConstraint.indicator_col, pd.Series(False, index=dataset.data.index)).to_numpy(dtype=np.int8).copy()
	stand_results = []
	progress = st.progress(0.0, text="Preparing stand optimisations")
	hyperparameters = batch_algorithm_hyperparameters()
	for position, stand in stands.iterrows():
		stand_id = stand["stand_id"]
		progress.progress(
			position / max(len(stands), 1),
			text=f"Optimising {stand_id} ({position + 1} of {len(stands)})",
		)
		stand_dataset = stand_dataset_from_global(dataset, stand_id, stand.geometry)
		if stand_dataset._n_trees == 0:
			stand_results.append({
				"stand_id": stand_id,
				"status": "empty",
				"tree_count": 0,
				"hectares": stand_dataset.hectares,
				"error": "No trees were assigned to this stand.",
			})
			continue
		try:
			original_basal_area, _, _ = stand_basal_area_metrics(stand_dataset)
			objectives = build_objectives(stand_dataset, obj_weights)
			constraints = build_constraints(stand_dataset)
			optimiser = build_optimiser(stand_dataset, objectives, constraints)
			result = optimiser.optimise(algorithm_hyperparameters=hyperparameters)
			validate_buffer_decisions(stand_dataset, result)
			_, retained_basal_area, retained_basal_area_percentage = stand_basal_area_metrics(
				stand_dataset, result["decision_vector"]
			)
			source_rows = stand_dataset.data["CG_source_row"].to_numpy(dtype=int)
			combined_vector[source_rows] = result["decision_vector"]
			stand_results.append({
				"stand_id": stand_id,
				"status": result.get("status", "unknown"),
				"tree_count": stand_dataset._n_trees,
				"cut_count": len(result.get("cut_indices", result.get("decisions", []))),
				"protected_count": int(stand_dataset.data[
					WaterProtectionConstraint.indicator_col
				].sum()),
				"hectares": stand_dataset.hectares,
				"original_basal_area_m2_ha": original_basal_area,
				"retained_basal_area_m2_ha": retained_basal_area,
				"retained_basal_area_percentage": retained_basal_area_percentage,
				"result": result,
				"dataset": stand_dataset,
				"objectives": objectives,
				"constraints": constraints,
				"optimiser": optimiser,
			})
		except Exception as exc:
			stand_results.append({
				"stand_id": stand_id,
				"status": "failed",
				"tree_count": stand_dataset._n_trees,
				"hectares": stand_dataset.hectares,
				"error": str(exc),
			})
	progress.progress(1.0, text="Stand optimisations complete")
	combined_result = {
		"status": "partial" if any(item.get("status") == "failed" for item in stand_results) else "completed",
		"decision_vector": combined_vector,
		"decisions": np.flatnonzero(combined_vector).tolist(),
		"cut_indices": np.flatnonzero(combined_vector).tolist(),
		"retained_indices": np.flatnonzero(1 - combined_vector).tolist(),
		"treatment_type": "thinning_treatment",
	}
	return combined_result, stand_results


def render_result_card(label: object, value: str) -> None:
	"""Render one consistently styled result card."""
	st.markdown(
		f'''
		<div class="result-card">
			<div class="result-label">{html.escape(str(label))}</div>
			<div class="result-value">{value}</div>
		</div>
		''',
		unsafe_allow_html=True,
	)

def render_objective_score_cards(scores: List[Dict[str, float]]) -> None:
	if not scores:
		return

	st.markdown("#### Objective diagnostics")
	columns = st.columns(len(scores))
	for index, score in enumerate(scores):
		with columns[index]:
			render_result_card(score["objective"], f'{float(score["score"]):.2f}')


def render_benchmark_score_cards(benchmarks: Dict[str, dict]) -> None:
	if not benchmarks:
		return

	st.markdown("#### Objective performance")
	columns = st.columns(len(benchmarks))
	for index, (objective_name, benchmark) in enumerate((benchmarks or {}).items()):
		percent = benchmark.get("percent")
		value = "N/A" if percent is None else f"{float(percent):.1f}%"
		with columns[index]:
			render_result_card(objective_name, value)


def render_timing_score_cards(results: dict) -> None:
	"""Render end-to-end and backend timing phases below the decision map."""
	algorithm_timing = results.get("timing") or {}
	timings = [
		("Total runtime", results.get("total_seconds")),
		(
			"Normalisation (calculating min/max ranges)",
			results.get("normalization_seconds"),
		),
		(
			"Algorithm setup: model and constraint preparation",
			algorithm_timing.get("setup_seconds"),
		),
		(
			"Optimization library call",
			algorithm_timing.get("engine_seconds"),
		),
	]
	timings = [item for item in timings if item[1] is not None]
	if not timings:
		return

	st.markdown("#### Timing")
	columns = st.columns(len(timings))
	for column, (label, value) in zip(columns, timings):
		with column:
			render_result_card(label, f"{float(value):.3f}s")

	engine_call = algorithm_timing.get("engine_call")
	if engine_call:
		st.caption(f"Engine call: {engine_call}")


def render_thinning_parameterisation(dataset: SingleTreeDataset, results: dict) -> None:
	"""Render post-optimisation thinning controls for future-crop selection."""
	if results.get("treatment_type") != "future_crop_tree_selection":
		return
	if not dataset.hectares or dataset.hectares <= 0:
		st.warning("A positive stand area is required to parameterise thinnings.")
		return

	st.markdown("#### Parameterise thinning treatments")
	target_metric = "Basal area" if st.toggle(
		"Use basal area targets",
		key="thinning_use_basal_area",
	) else "Density"
	st.caption(
		"Drag each circle vertically to set the value remaining after that "
		"treatment. Targets cannot increase between treatments or fall below "
		"the future-crop-tree value."
	)
	n_thinnings = st.select_slider(
		"Number of thinning treatments",
		options=[1, 2, 3, 4, 5],
		key="n_thinning_treatments",
	)
	basal_areas = tree_basal_areas_per_hectare(dataset)
	future_crop_indices = results.get("future_crop_tree_indices", [])
	if target_metric == "Basal area":
		current_value = float(basal_areas.sum())
		future_crop_value = float(basal_areas[future_crop_indices].sum())
		unit_label = "m2/ha"
		value_step = 0.1
		targets_key = "thinning_basal_area_targets"
	else:
		current_value = dataset._n_trees / dataset.hectares
		future_crop_value = len(future_crop_indices) / dataset.hectares
		unit_label = "stems/ha"
		value_step = 1.0
		targets_key = "thinning_density_targets"
	previous_targets = st.session_state.get(targets_key, [])
	treatment_count_changed = len(previous_targets) != n_thinnings
	if len(previous_targets) != n_thinnings:
		previous_targets = [current_value] * n_thinnings
	if treatment_count_changed or not previous_targets:
		previous_targets = [
			current_value - (current_value - future_crop_value) * index / n_thinnings
			for index in range(1, n_thinnings + 1)
		]

	targets = thinning_schedule_chart(
		n_thinnings=n_thinnings,
		current_value=current_value,
		future_crop_value=future_crop_value,
		unit_label=unit_label,
		value_step=value_step,
		default_targets=previous_targets,
		key=f"thinning_schedule_chart_{n_thinnings}",
	)
	if treatment_count_changed or targets != previous_targets:
		st.session_state.thinning_schedule = None
	st.session_state[targets_key] = targets

	if st.button("Create thinning schedule", use_container_width=True):
		if any(
			previous < following
			for previous, following in zip(targets, targets[1:])
		):
			st.error(
				f"Remaining {unit_label} cannot increase between thinning treatments. "
				f"Move each later treatment to the same or a lower {unit_label}."
			)
			return
		optimiser = st.session_state.get("optimisation_optimiser")
		if optimiser is None:
			st.error("The optimiser for this result is no longer available. Run the optimisation again.")
			return
		try:
			# Calling through the current class also supports optimiser instances
			# retained by Streamlit across a source-code hot reload.
			optimiser.future_crop_tree_indices = results.get(
				"future_crop_tree_indices", []
			)
			optimiser.calculate_competition_indices = (
				FutureCropTreeOptimiser.calculate_competition_indices.__get__(optimiser)
			)
			if target_metric == "Basal area":
				st.session_state.thinning_schedule = FutureCropTreeOptimiser.thinning_schedule(
					optimiser, targets, target_metric="basal_area"
				)
			else:
				stem_targets = [round(target * dataset.hectares) for target in targets]
				st.session_state.thinning_schedule = FutureCropTreeOptimiser.thinning_schedule(
					optimiser, stem_targets
				)
		except Exception as exc:
			st.error(f"Unable to create thinning schedule: {exc}")
			return
		st.success("Thinning schedule created.")

	if st.session_state.get("thinning_schedule"):
		st.checkbox(
			"Show decision map",
			key="show_thinning_decision_map",
			help="Show future crop trees and each thinning treatment in separate colours.",
		)
		if st.session_state.show_thinning_decision_map:
			st.checkbox(
				"Show nearest Z tree",
				key="show_nearest_z_tree",
				help="Draw a line from every non-Z tree to its nearest future crop tree.",
			)
			optimiser = st.session_state.optimisation_optimiser
			decision_groups = {
				"Future crop trees": results.get("future_crop_tree_indices", []),
				**{
					f"Thinning {item['thinning_round']}": item["tree_indices"]
					for item in st.session_state.thinning_schedule
				},
			}
			render_pydeck_decision_map(
				dataset.data,
				dataset._get_column_name("x"),
				dataset._get_column_name("y"),
				dataset.epsg,
				results.get("future_crop_tree_indices", []),
				dataset.columns.get("id"),
				dataset.columns.get("dbh"),
				dataset.columns.get("species"),
				decision_groups=decision_groups,
				competition_indices=optimiser.competition_indices,
				nearest_z_tree_indices=optimiser.nearest_z_tree_indices,
				show_nearest_z_tree=st.session_state.show_nearest_z_tree,
				selected_tree_circle_radius=(
					float(st.session_state.constraint_min_distance_value) / 2.0
					if st.session_state.constraint_min_distance else 0.0
				),
			)
		summary = pd.DataFrame([
			{
				"Treatment": item["thinning_round"],
				f"Remaining {unit_label} target": targets[item["thinning_round"] - 1],
				"Trees assigned": len(item["tree_indices"]),
			}
			for item in st.session_state.thinning_schedule
		])
		st.dataframe(summary, hide_index=True, use_container_width=True)


def render_frequency_controls(dataset: SingleTreeDataset, results: dict) -> None:
	"""Render editable post-optimisation species and DBH frequencies."""
	if not dataset.columns.get("species") or not dataset.columns.get("dbh"):
		return
	decision_vector = np.asarray(results.get("decision_vector", []), dtype=bool)
	if decision_vector.size != dataset._n_trees:
		return
	desired_mask = (
		decision_vector
		if results.get("treatment_type") == "future_crop_tree_selection"
		else ~decision_vector
	)

	st.markdown("#### Desired post-optimisation frequencies")
	species_column = dataset._get_column_name("species")
	species_values = dataset.data[species_column].astype("string")
	species_labels = sorted(species_values.dropna().unique().tolist())
	species_actual = [int((species_values == label).sum()) for label in species_labels]
	species_desired = [
		int(((species_values == label).to_numpy() & desired_mask).sum())
		for label in species_labels
	]
	with st.container(border=True):
		st.session_state.desired_species_frequency = frequency_histogram(
			species_labels,
			species_actual,
			st.session_state.get("desired_species_frequency") or species_desired,
			"Species",
			key="desired-species-frequency",
		)

	dbh_column = dataset._get_column_name("dbh")
	dbh_values = pd.to_numeric(dataset.data[dbh_column], errors="coerce").to_numpy(dtype=float)
	finite = np.isfinite(dbh_values) & (dbh_values >= 0)
	if not finite.any():
		return
	max_dbh = float(dbh_values[finite].max())
	dbh_edges = np.arange(0.0, 5.0 * np.ceil(max_dbh / 5.0) + 5.0, 5.0)
	dbh_labels = [f"{int(left)}-{int(right)}" for left, right in zip(dbh_edges[:-1], dbh_edges[1:])]
	dbh_actual, _ = np.histogram(dbh_values[finite], bins=dbh_edges)
	dbh_desired, _ = np.histogram(dbh_values[finite & desired_mask], bins=dbh_edges)
	with st.container(border=True):
		st.session_state.desired_dbh_frequency = frequency_histogram(
			dbh_labels,
			dbh_actual.tolist(),
			st.session_state.get("desired_dbh_frequency") or dbh_desired.tolist(),
			"DBH (cm)",
			key="desired-dbh-frequency",
		)


def decision_histogram(
	values: pd.Series,
	cut_mask: np.ndarray,
	bin_edges: np.ndarray,
	x_axis_title: str,
	decision_labels: tuple[str, str] = ("Retain", "Cut"),
) -> go.Figure:
	"""Build an overlaid retained/cut histogram normalized within each decision."""
	numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
	finite = np.isfinite(numeric)
	figure = go.Figure()
	for label, decision, colour in (
		(decision_labels[0], False, "#009E73"),
		(decision_labels[1], True, "#D55E00"),
	):
		group_values = numeric[finite & (cut_mask == decision)]
		counts, _ = np.histogram(group_values, bins=bin_edges)
		proportions = counts / counts.sum() if counts.sum() else counts.astype(float)
		figure.add_bar(
			x=(bin_edges[:-1] + bin_edges[1:]) / 2,
			y=proportions,
			width=np.diff(bin_edges),
			name=label,
			marker_color=colour,
			opacity=0.55,
			hovertemplate=f"{label}<br>{x_axis_title}: %{{x:.2f}}<br>Proportion: %{{y:.1%}}<extra></extra>",
		)
	figure.update_layout(
		barmode="overlay",
		xaxis_title=x_axis_title,
		yaxis_title="Proportion of trees",
		yaxis_tickformat=".0%",
		legend_title_text="Decision",
		margin=dict(l=20, r=20, t=30, b=20),
	)
	return figure


def species_decision_histogram(
	values: pd.Series,
	cut_mask: np.ndarray,
	decision_labels: tuple[str, str] = ("Retain", "Cut"),
) -> go.Figure:
	"""Build an overlaid current species frequency histogram."""
	labels = sorted(values.dropna().astype(str).unique().tolist())
	figure = go.Figure()
	for label, decision, colour in (
		(decision_labels[0], False, "#009E73"),
		(decision_labels[1], True, "#D55E00"),
	):
		counts = [int(((values.astype(str) == species) & (cut_mask == decision)).sum()) for species in labels]
		figure.add_bar(x=labels, y=counts, name=label, marker_color=colour, opacity=0.55)
	figure.update_layout(
		barmode="overlay", xaxis_title="Species", yaxis_title="Number of trees",
		legend_title_text="Decision", margin=dict(l=20, r=20, t=30, b=20),
	)
	return figure


def volume_histogram_edges(values: pd.Series) -> np.ndarray:
	"""Return shared volume bins with a fixed width of 0.25 m3."""
	numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
	numeric = numeric[np.isfinite(numeric)]
	if numeric.size == 0:
		return np.array([0.0, 0.25])
	bin_start = 0.25 * np.floor(float(numeric.min()) / 0.25)
	bin_end = 0.25 * np.ceil(float(numeric.max()) / 0.25)
	if bin_end <= bin_start:
		bin_end = bin_start + 0.25
	return np.arange(bin_start, bin_end + 0.25, 0.25)


def render_stand_optimisation_results(
	dataset: SingleTreeDataset,
	results: dict,
	stand_results: list[dict],
) -> None:
	"""Render combined and per-stand thinning results."""
	st.markdown("<div class='step-title'>5. Results by stand</div>", unsafe_allow_html=True)
	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	summary_rows = []
	for item in stand_results:
		retained_basal_area = item.get("retained_basal_area_m2_ha")
		retained_percentage = item.get("retained_basal_area_percentage")
		basal_area_display = (
			f"{float(retained_basal_area):.2f} [{float(retained_percentage):.1f}%]"
			if retained_basal_area is not None and retained_percentage is not None
			else "N/A"
		)
		summary_rows.append({
			"Stand": item["stand_id"],
			"Status": item["status"],
			"Area (ha)": round(float(item.get("hectares", 0.0)), 4),
			"Trees": int(item.get("tree_count", 0)),
			"Trees cut": int(item.get("cut_count", 0)),
			"Basal area kept (m²/ha) [%]": basal_area_display,
			"Water-protected": int(item.get("protected_count", 0)),
			"Error": item.get("error", ""),
		})
	summary = pd.DataFrame(summary_rows)
	stand_query = st.text_input(
		"Filter diagnostics by stand ID",
		key="diagnostic_stand_id",
		placeholder="e.g. stand_023",
		help="Leave blank to show all stands. The entered ID must match a result ID exactly.",
	)
	selected_stand_id = stand_query if stand_query else None
	available_stand_ids = set(summary["Stand"].astype(str))
	if stand_query.strip() and selected_stand_id not in available_stand_ids:
		st.warning(f"No stand matches {stand_query!r}; diagnostics are showing all stands.")
		selected_stand_id = None

	if selected_stand_id is None:
		diagnostic_mask = np.ones(dataset._n_trees, dtype=bool)
		filtered_summary = summary
	else:
		diagnostic_mask = dataset.data["stand_id"].eq(selected_stand_id).to_numpy(dtype=bool)
		filtered_summary = summary.loc[summary["Stand"].eq(selected_stand_id)]
	diagnostic_data = dataset.data.loc[diagnostic_mask].copy().reset_index(drop=True)
	diagnostic_vector = np.asarray(results["decision_vector"], dtype=np.int8)[diagnostic_mask]
	diagnostic_decisions = np.flatnonzero(diagnostic_vector).tolist()

	completed = sum("result" in item for item in stand_results)
	failed = len(stand_results) - completed
	metric_columns = st.columns(4)
	metric_columns[0].metric("Stands", len(summary))
	metric_columns[1].metric("Completed", completed)
	metric_columns[2].metric("Failed", failed)
	metric_columns[3].metric("Trees cut", len(results.get("decisions", [])))
	st.dataframe(filtered_summary, hide_index=True, use_container_width=True)

	eligible = buffer_eligible_mask(diagnostic_data)
	frequency_data = diagnostic_data.loc[eligible].reset_index(drop=True)
	frequency_vector = diagnostic_vector[eligible]
	st.caption("Distributions exclude road-clearing and water-protected trees; harvest totals include road clearing.")
	st.markdown("#### Thinning diagnostics")
	histogram_columns = st.columns(2)
	dbh_column = dataset.columns.get("dbh")
	with histogram_columns[0]:
		if dbh_column and dbh_column in frequency_data.columns and not frequency_data.empty:
			dbh_values = pd.to_numeric(frequency_data[dbh_column], errors="coerce").dropna()
			if dbh_values.empty:
				st.info("No numeric DBH values are available for this selection.")
			else:
				bin_start = 5.0 * np.floor(float(dbh_values.min()) / 5.0)
				bin_end = 5.0 * np.ceil(float(dbh_values.max()) / 5.0)
				if bin_end <= bin_start:
					bin_end = bin_start + 5.0
				dbh_edges = np.arange(bin_start, bin_end + 5.0, 5.0)
				st.plotly_chart(
					decision_histogram(
						frequency_data[dbh_column], frequency_vector.astype(bool),
						dbh_edges, "DBH (cm)",
					),
					use_container_width=True,
					key=f"dbh-diagnostic-{selected_stand_id or 'all'}",
				)
		else:
			st.info("Map a DBH field to display the DBH histogram.")

	species_column = dataset.columns.get("species")
	with histogram_columns[1]:
			if species_column and species_column in frequency_data.columns and not frequency_data.empty:
				st.plotly_chart(
					species_decision_histogram(
						frequency_data[species_column], frequency_vector.astype(bool),
					),
					use_container_width=True,
					key=f"species-diagnostic-{selected_stand_id or 'all'}",
				)
			else:
				st.info("Map a species field to display the species histogram.")



	st.checkbox("Show combined decision map", key="show_decision_map")
	if st.session_state.show_decision_map:
		stand_bytes = st.session_state.get("stand_shapefile_bytes")
		water_bytes = getattr(dataset, "buffer_inputs", st.session_state).get("water_shapefile_bytes")
		layer_columns = st.columns(2)
		with layer_columns[0]:
			st.checkbox(
				"Stand boundaries",
				key="show_stand_boundaries_layer",
				disabled=not bool(stand_bytes),
				help="Overlay the uploaded stand borders on the decision map.",
			)
		with layer_columns[1]:
			st.checkbox(
				"Water bodies and buffer",
				key="show_water_bodies_layer",
				disabled=not bool(water_bytes),
				help="Overlay uploaded water bodies and the configured protection buffer.",
			)
		map_stands = (
			read_geometry_shapefile(stand_bytes, dataset.epsg)
			if stand_bytes and st.session_state.show_stand_boundaries_layer else None
		)
		if map_stands is not None and selected_stand_id is not None:
			map_stands = map_stands.reset_index(drop=True)
			map_stands["stand_id"] = [
				f"stand_{index + 1:03d}" for index in range(len(map_stands))
			]
			map_stands = map_stands.loc[map_stands["stand_id"].eq(selected_stand_id)]
		map_water = (
			read_geometry_shapefile(water_bytes, dataset.epsg)
			if water_bytes and st.session_state.show_water_bodies_layer else None
		)
		render_pydeck_decision_map(
			diagnostic_data,
			dataset._get_column_name("x"),
			dataset._get_column_name("y"),
			dataset.epsg,
			diagnostic_decisions,
			dataset.columns.get("id"),
			dataset.columns.get("dbh"),
			dataset.columns.get("species"),
			"thinning_treatment",
			stand_boundaries=map_stands,
			water_bodies=map_water,
			water_buffer_distance=getattr(dataset, "buffer_inputs", st.session_state).get("water_buffer_distance", 0.0),
			roads=render_roads_layer_control(dataset),
			road_buffer_distance=getattr(dataset, "buffer_inputs", st.session_state).get("road_buffer_distance", 0.0),
		)
	validation = st.session_state.get("spatial_validation") or {}
	if validation.get("boundary_ties"):
		st.caption(
			f"Randomly assigned {validation['boundary_ties']:,} tree(s) that "
			"intersected more than one stand."
		)
	if validation.get("unassigned_trees"):
		st.caption(
			f"Retained {validation['unassigned_trees']:,} tree(s) outside all stands; "
			"they were not included in a stand optimisation."
		)
	st.markdown("</div>", unsafe_allow_html=True)


def render_result_diagnostics(dataset: SingleTreeDataset, results: dict) -> None:
	"""Render DBH and species decision distributions for any treatment."""
	decision_labels = (
		("Not selected", "Selected")
		if results.get("treatment_type") == "future_crop_tree_selection"
		else ("Retain", "Cut")
	)
	decision_vector = np.asarray(results.get("decision_vector", []), dtype=bool)
	if decision_vector.size != dataset._n_trees:
		return
	eligible = buffer_eligible_mask(dataset.data)
	data = dataset.data.loc[eligible].reset_index(drop=True)
	decision_vector = decision_vector[eligible]
	st.caption("Distributions exclude road-clearing and water-protected trees; harvest totals include road clearing.")
	dbh_column = dataset.columns.get("dbh")
	species_column = dataset.columns.get("species")
	columns = st.columns(2)
	with columns[0]:
		if dbh_column and dbh_column in data.columns:
			values = pd.to_numeric(data[dbh_column], errors="coerce")
			finite = values.notna().to_numpy()
			if finite.any():
				start = 5.0 * np.floor(float(values[finite].min()) / 5.0)
				end = max(start + 5.0, 5.0 * np.ceil(float(values[finite].max()) / 5.0))
				st.plotly_chart(decision_histogram(
					values, decision_vector, np.arange(start, end + 5.0, 5.0), "DBH (cm)", decision_labels),
					use_container_width=True, key="dbh-result-diagnostic")
	with columns[1]:
		if species_column and species_column in data.columns:
			st.plotly_chart(species_decision_histogram(
				data[species_column], decision_vector, decision_labels),
				use_container_width=True, key="species-result-diagnostic")


def render_optimisation_results() -> None:
	results = st.session_state.get("optimisation_results")
	result_algorithm = st.session_state.get("optimisation_algorithm")
	dataset = st.session_state.get("optimisation_dataset")
	objectives = st.session_state.get("optimisation_objectives")
	benchmarks = st.session_state.get("optimisation_benchmarks")
	report_percentages = st.session_state.get("optimisation_report_percentages", False)
	constraints = st.session_state.get("optimisation_constraints")
	stand_results = st.session_state.get("optimisation_stand_results")
	if results and dataset is not None and stand_results is not None:
		render_stand_optimisation_results(dataset, results, stand_results)
		return
	if not results or dataset is None or not objectives:
		return
	st.markdown("<div class='step-title'>5. Results</div>", unsafe_allow_html=True)
	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	st.write("Optimiser status:", results["status"])
	if result_algorithm in {"dual_annealing", "genetic_algorithm"}:
		algorithm_label = (
			"simulated annealing" if result_algorithm == "dual_annealing"
			else "the genetic algorithm"
		)
		if not constraints:
			st.info("No constraints were configured for this run.")
		elif results.get("is_feasible", False):
			st.success("Feasible solution found: all configured constraints are satisfied.")
		else:
			violation = results.get("constraint_violation")
			violation_text = (
				"unknown" if violation is None else f"{float(violation):.6g}"
			)
			st.warning(
				f"No feasible solution was found during {algorithm_label}. "
				f"Aggregate constraint violation: {violation_text}."
			)
	if report_percentages:
		render_benchmark_score_cards(benchmarks)
	else:
		render_objective_diagnostics(list(objectives.keys()), results["decisions"])
	st.markdown("#### Decision diagnostics")
	render_result_diagnostics(dataset, results)
	st.checkbox(
		"Show decision map",
		key="show_decision_map",
		help="Render a map of the optimisation decisions.",
	)
	if st.session_state.show_decision_map:
		st.markdown("#### Decision map")
		water_bytes = getattr(dataset, "buffer_inputs", st.session_state).get("water_shapefile_bytes")
		st.checkbox(
			"Water bodies and buffer",
			key="show_water_bodies_layer",
			disabled=not bool(water_bytes),
			help="Overlay uploaded water bodies and the configured protection buffer.",
		)
		map_water = (
			read_geometry_shapefile(water_bytes, dataset.epsg)
			if water_bytes and st.session_state.show_water_bodies_layer else None
		)
		render_pydeck_decision_map(
			dataset.data,
			dataset._get_column_name("x"),
			dataset._get_column_name("y"),
			dataset.epsg,
			results["decisions"],
			dataset.columns.get("id"),
			dataset.columns.get("dbh"),
			dataset.columns.get("species"),
			results.get("treatment_type", "future_crop_tree_selection"),
			water_bodies=map_water,
			water_buffer_distance=getattr(dataset, "buffer_inputs", st.session_state).get("water_buffer_distance", 0.0),
			roads=render_roads_layer_control(dataset),
			road_buffer_distance=getattr(dataset, "buffer_inputs", st.session_state).get("road_buffer_distance", 0.0),
			selected_tree_circle_radius=(
				float(st.session_state.constraint_min_distance_value) / 2.0
				if (
					results.get("treatment_type") == "future_crop_tree_selection"
					and st.session_state.constraint_min_distance
				) else 0.0
			),
		)
	render_thinning_parameterisation(dataset, results)
	render_timing_score_cards(results)
	st.markdown("</div>", unsafe_allow_html=True)


def polygon_from_folium_output(map_output: Optional[dict]):
	if not isinstance(map_output, dict):
		return None

	drawings = map_output.get("all_drawings") or []
	if not drawings and map_output.get("last_active_drawing"):
		drawings = [map_output["last_active_drawing"]]

	for drawing in reversed(drawings):
		geometry = drawing.get("geometry") if isinstance(drawing, dict) else None
		if isinstance(geometry, dict) and geometry.get("type") in {"Polygon", "MultiPolygon"}:
			return shape(geometry)
	return None


def render_folium_map(df: pd.DataFrame, x_col: str, y_col: str, epsg_code: int) -> pd.DataFrame:
	map_data = prepare_map_data(df, x_col, y_col, epsg_code)
	if map_data is None:
		st.warning("No valid coordinate rows were found for the map.")
		return df
	gdf, gdf_wgs84 = map_data

	# Keep the full original row data (all columns) aligned to the filtered coords
	full_data = df.loc[gdf.index]
	center_lat = float(gdf_wgs84.geometry.y.mean())
	center_lon = float(gdf_wgs84.geometry.x.mean())
	m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")
	Draw(
		export=False,
		draw_options={
			"polyline": False,
			"rectangle": True,
			"circle": False,
			"circlemarker": False,
			"marker": False,
			"polygon": True,
		},
		edit_options={"edit": True, "remove": True},
	).add_to(m)

	for idx, row in gdf_wgs84.iterrows():
		data_row = full_data.loc[idx]
		tooltip_html = "<br>".join(
			f"{col}: {data_row[col]}" for col in full_data.columns
		)
		folium.CircleMarker(
			location=[row.geometry.y, row.geometry.x],
			radius=4,
			color="#1e5f4f",
			fill=True,
			fill_opacity=0.75,
			tooltip=folium.Tooltip(tooltip_html),
		).add_to(m)

	if st_folium is not None:
		map_output = st_folium(
			m,
			width=None,
			height=360,
			returned_objects=["all_drawings", "last_active_drawing"],
			key="tree_location_filter_map",
		)
		polygon = polygon_from_folium_output(map_output)
		if polygon is None:
			st.caption("Draw a polygon or rectangle on the map to filter the uploaded trees.")
			return df

		polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326").to_crs(gdf.crs)
		mask = gdf.geometry.intersects(polygon_gdf.geometry.iloc[0])
		filtered_df = df.loc[gdf.index[mask]].copy().reset_index(drop=True)
		if filtered_df.empty:
			st.warning("The drawn polygon does not contain any trees. The full dataset is still active.")
			return df
		st.success(f"Polygon filter selected {len(filtered_df):,} of {len(df):,} trees.")
		return filtered_df
	elif components is not None:
		components.html(m._repr_html_(), height=380, scrolling=False)
	else:
		st.warning("Folium map preview is unavailable in this environment.")

	return df


@st.cache_data
def image_to_data_uri(image_path: Path) -> str:
	mime_types = {
		".png": "image/png",
		".jpg": "image/jpeg",
		".jpeg": "image/jpeg",
		".svg": "image/svg+xml",
	}
	image_bytes = image_path.read_bytes()
	encoded = base64.b64encode(image_bytes).decode("ascii")
	mime_type = mime_types.get(image_path.suffix.lower(), "application/octet-stream")
	return f"data:{mime_type};base64,{encoded}"


def render_choice_card(
	*,
	title: str,
	subtitle: str,
	image_path: Path,
	disabled: bool = False,
	active: bool = False,
) -> None:
	card_classes = ["choice-card"]
	if active:
		card_classes.append("active")
	if disabled:
		card_classes.append("disabled")
	description = f'<div class="choice-description">{html.escape(subtitle)}</div>' if subtitle else ""
	image_class = "choice-image disabled" if disabled else "choice-image"
	st.markdown(
		f'''
		<div class="{' '.join(card_classes)}">
			<div class="{image_class}">
				<img src="{image_to_data_uri(image_path)}" alt="{html.escape(title)}" />
			</div>
			<div class="choice-title">{html.escape(title)}</div>
			{description}
		</div>
		''',
		unsafe_allow_html=True,
	)


def validation_report(mapping: Dict[str, Optional[str]]) -> List[str]:
	"""Report only mappings required by the current configuration."""
	required = {"id", "x_coord", "y_coord"}
	if is_thinning_treatment():
		if st.session_state.objective_min_volume:
			required.add("volume")
		if st.session_state.constraint_basal_area or st.session_state.specify_dbh_distribution:
			required.add("dbh")
		if st.session_state.specify_species_distribution:
			required.add("species")
		if st.session_state.specify_dbh_distribution:
			required.add("dbh")
	else:
		if st.session_state.objective_social_status:
			required.add("social_status")
		if st.session_state.objective_wood_quality:
			required.add("wood_quality")
		if st.session_state.objective_min_dbh:
			required.add("dbh")
		if st.session_state.constraint_only_living_trees:
			required.add("is_alive")
		if st.session_state.specify_species_distribution:
			required.add("species")
		if st.session_state.specify_dbh_distribution:
			required.add("dbh")
	missing = []
	for role in sorted(required):
		if not mapping.get(role):
			missing.append(role.replace("_", " "))
	return missing


def render_spatial_treatment_inputs() -> None:
	"""Use only bundled spatial layers belonging to the selected case study."""
	for prefix in ("stand_shapefile", "water_shapefile", "roads_gpkg"):
		st.session_state[f"{prefix}_bytes"] = None
		st.session_state[f"{prefix}_name"] = None
	if st.session_state.get("data_source") != "evo":
		return
	preset = TEST_DATASETS["evo"]
	try:
		water_bytes = (TOOL_DIR / "data/examples" / preset["water_file"]).read_bytes()
		roads_bytes = (TOOL_DIR / "data/examples" / preset["roads_file"]).read_bytes()
		water = read_geometry_shapefile(water_bytes, int(preset["epsg"]))
		roads = read_roads_geopackage(roads_bytes)
		st.session_state.water_shapefile_bytes = water_bytes
		st.session_state.water_shapefile_name = Path(preset["water_file"]).name
		st.session_state.roads_gpkg_bytes = roads_bytes
		st.session_state.roads_gpkg_name = Path(preset["roads_file"]).name
		st.caption(f"Included case study layers: {len(water):,} water bodies and {len(roads):,} road features.")
	except Exception:
		st.error("The Evo water and road layers could not be loaded. Please contact the dashboard maintainer.")
		st.stop()


def render_problem_section() -> None:
	"""Render the silvicultural-problem selection section."""
	st.markdown("<div class='step-title'>2. Silvicultural problem</div>", unsafe_allow_html=True)
	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	visible_problems = {key: PROBLEM_METADATA[key] for key in ("future_crop_tree_selection", "thinning_treatment")}
	compatible_problem = TEST_DATASETS.get(st.session_state.get("data_source"), {}).get("problem")
	problem_cols = st.columns(len(visible_problems))
	for idx, (problem_key, problem) in enumerate(visible_problems.items()):
		disabled = problem_key != compatible_problem
		with problem_cols[idx]:
			render_choice_card(
				title=problem["title"],
				subtitle="",
				image_path=IMAGE_DIR / problem["image"],
				disabled=disabled,
				active=st.session_state.selected_problem == problem_key,
			)
			st.button(
				"Select",
				key=f"problem_{problem_key}",
				type="primary" if st.session_state.selected_problem == problem_key else "secondary",
				use_container_width=True,
				disabled=disabled,
				on_click=select_problem if not disabled else None,
				args=(problem_key,) if not disabled else None,
			)
	render_spatial_treatment_inputs()
	st.markdown("</div>", unsafe_allow_html=True)


def render_algorithm_choices() -> None:
	"""Render algorithm cards backed by the central algorithm registry."""
	visible_algorithms = {"linear_programming": ALGORITHM_METADATA["linear_programming"]}
	algorithm_cols = st.columns(len(visible_algorithms))
	for idx, (algorithm_key, algorithm) in enumerate(visible_algorithms.items()):
		disabled = not algorithm["supported"]
		with algorithm_cols[idx]:
			render_choice_card(
				title=algorithm["title"],
				subtitle=algorithm["subtitle"],
				image_path=IMAGE_DIR / algorithm["image"],
				disabled=disabled,
				active=st.session_state.selected_algorithm == algorithm_key,
			)
			st.button(
				"Select",
				key=f"algorithm_{algorithm_key}",
				type="primary" if st.session_state.selected_algorithm == algorithm_key else "secondary",
				use_container_width=True,
				disabled=disabled,
				on_click=set_state_value if not disabled else None,
				args=("selected_algorithm", algorithm_key) if not disabled else None,
			)


def render_objective_weights() -> dict[str, float]:
	"""Render objective weights and return the current normalized mapping."""
	selected_flags = [
		st.session_state.objective_social_status,
		st.session_state.objective_wood_quality,
		st.session_state.objective_min_dbh,
	]
	selected = [
		objective
		for objective, enabled in zip(SUPPORTED_OBJECTIVES, selected_flags)
		if enabled
	]
	selected_count = len(selected)
	selected_key = tuple(selected)

	if selected_count > 1:
		st.markdown("##### Objective weights")
		if st.session_state.get("_last_selected_objectives") != selected_key:
			default_weight = 1.0 / selected_count
			for index, objective in enumerate(selected):
				st.session_state[f"weight_{objective}"] = (
					1.0 - default_weight * (selected_count - 1)
					if index == selected_count - 1
					else default_weight
				)
			st.session_state["_last_selected_objectives"] = selected_key

		default_weights = {
			objective: float(st.session_state.get(
				f"weight_{objective}", 1.0 / selected_count
			))
			for objective in selected
		}
		weights = objective_weight_slider(
			objectives=selected,
			labels=OBJECTIVE_LABELS,
			default_weights=default_weights,
			key="objective_weight_slider",
		)
		for objective, weight in weights.items():
			st.session_state[f"weight_{objective}"] = weight

		return weights

	if selected_count == 1:
		st.info("Only one objective is active, so its weight is fixed at 1.")
		return {selected[0]: 1.0}

	st.warning("No objectives selected. Map the required objective columns, then select at least one objective.")
	return {}


def select_data_source(source: Optional[str]) -> None:
	"""Switch inventories without carrying over mappings, layers or results."""
	if source not in TEST_DATASETS:
		source = None
	st.session_state.data_source = source
	st.session_state.column_mapping = {}
	st.session_state.epsg_text = ""
	for key in list(st.session_state):
		if key.startswith(("optimisation_", "desired_", "thinning_")) or key in {
			"id_column", "x_column", "y_column", "dbh_column", "species_column",
			"social_status_column", "is_alive_column", "wood_quality_column", "volume_column",
			"stand_shapefile_upload", "water_shapefile_upload", "roads_geopackage_upload",
			"stand_shapefile_bytes", "water_shapefile_bytes", "roads_gpkg_bytes",
			"stand_shapefile_name", "water_shapefile_name", "roads_gpkg_name", "spatial_validation",
		}:
			st.session_state.pop(key, None)
	# Set widget values explicitly: deleting keys can restore browser-side selections.
	for key in ("id_column", "x_column", "y_column", "dbh_column", "species_column",
		"social_status_column", "is_alive_column", "wood_quality_column", "volume_column"):
		st.session_state[key] = "Not set"
	for key in ("objective_social_status", "objective_wood_quality", "objective_min_dbh", "objective_min_volume"):
		st.session_state[key] = False
	if source in TEST_DATASETS:
		preset = TEST_DATASETS[source]
		st.session_state.epsg_text = preset["epsg"]
		select_problem(preset["problem"])
	else:
		st.session_state.selected_problem = None


def render_upload_section() -> Optional[pd.DataFrame]:
	st.markdown("<div class='step-title'>0. Select a case study dataset</div>", unsafe_allow_html=True)
	case_columns = st.columns(2)
	for column, (key, preset) in zip(case_columns, TEST_DATASETS.items()):
		with column:
			st.button(preset["label"], key=f"example_{key}", on_click=select_data_source,
				args=(key,), use_container_width=True)
			st.markdown(
				f'<div class="case-study-map"><img src="{image_to_data_uri(IMAGE_DIR / preset["image"])}" '
				f'alt="{html.escape(preset["image_alt"])}" /></div>',
				unsafe_allow_html=True,
			)

	df: Optional[pd.DataFrame] = None
	source = st.session_state.get("data_source", "upload")
	if source in TEST_DATASETS:
		preset = TEST_DATASETS[source]
		try:
			df = read_csv_bytes((TOOL_DIR / "data/examples" / preset["file"]).read_bytes())
		except Exception as exc:
			st.error(f"Unable to load the test dataset: {exc}")

	st.markdown("<div class='step-title'>1. Map attributes</div>", unsafe_allow_html=True)

	if df is not None:
		st.markdown("<div class='step-card'>", unsafe_allow_html=True)

		left, right = st.columns([1, 1.08])
		with right:
			st.caption("Data preview")
			st.dataframe(df.head(5), use_container_width=True)

		columns = [str(column) for column in df.columns.tolist()]
		if columns:
			role_options = ["Not set"] + columns
			with left:
				map_rows = [st.columns(4) for _ in range(3)]
				field_specs = [
					("id", "**Tree ID (required)**", "id_column"),
					("x_coord", "**X coordinate (required)**", "x_column"),
					("y_coord", "**Y coordinate (required)**", "y_column"),
					("dbh", "DBH [cm]", "dbh_column"),
					("species", "Species", "species_column"),
					("social_status", "Social status", "social_status_column"),
					("is_alive", "Alive", "is_alive_column"),
					("wood_quality", "Wood quality", "wood_quality_column"),
					("volume", "Volume", "volume_column"),
				]
				for index, (field, label, key) in enumerate(field_specs):
					row = index // 4
					col = index % 4
					with map_rows[row][col]:
						current_value = st.session_state.column_mapping.get(field)
						if current_value is not None and not isinstance(current_value, str):
							current_value = None
						st.session_state.column_mapping[field] = st.selectbox(
							label,
							role_options,
							index=role_options.index(current_value) if current_value in role_options else 0,
							key=key,
						)
						if st.session_state.column_mapping[field] == "Not set":
							st.session_state.column_mapping[field] = None

				placeholder_labels = [
					"Health", "Biodiversity value", "Bark beetle risk",
					"Height of lowest living branch", "Stem defects", "Whorl positions",
					"Branch size", "Cavities", "Crown structure complexity",
				]
				st.markdown("##### Planned attributes (WP1)")
				placeholder_rows = [st.columns(3) for _ in range(3)]
				for index, label in enumerate(placeholder_labels):
					with placeholder_rows[index // 3][index % 3]:
						st.markdown(
							f"**{html.escape(label)}**<br>"
							"<span style='color:#777'>To be implemented</span>",
							unsafe_allow_html=True,
						)


		st.markdown("</div>", unsafe_allow_html=True)
	else:
		st.info("Select a case study dataset to continue.")

	return df


def render_objectives_constraints_section(
	df: Optional[pd.DataFrame],
) -> dict[str, float]:
	st.markdown("<div class='step-title'>3. Objectives and constraints</div>", unsafe_allow_html=True)
	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	multi_stand = bool(st.session_state.get("stand_shapefile_bytes"))
	if multi_stand:
		# Clear active options before creating widgets so disabled constraints cannot run.
		for key in ("specify_species_distribution", "specify_dbh_distribution", "constraint_basal_area"):
			st.session_state[key] = False

	dbh_column = st.session_state.column_mapping.get("dbh")
	dbh_values = (
		pd.to_numeric(df[dbh_column], errors="coerce").dropna()
		if df is not None and isinstance(dbh_column, str) and dbh_column in df.columns
		else pd.Series(dtype=float)
	)
	if is_thinning_treatment():
		volume_column = st.session_state.column_mapping.get("volume")

		thinning_cols = st.columns(4)
		with thinning_cols[0]:
			st.subheader("Economic return")
			st.checkbox(
				(
					"Maximise volume of selected trees"
					if is_thinning_from_above() else "Minimise volume of selected trees"
				),
				key="objective_min_volume",
				disabled=not bool(volume_column),
				help=(
					'Prioritise high-volume trees for removal.\n\nRequires the `volume` column.'
					if is_thinning_from_above()
					else 'Prioritise low-volume trees for removal.\n\nRequires the `volume` column.'
				),
			)
		with thinning_cols[1]:
			st.subheader("Biodiversity")
			st.number_input(
				"Water protection distance (m)",
				min_value=0.0,
				step=1.0,
				key="water_buffer_distance",
				disabled=not bool(st.session_state.get("water_shapefile_bytes")),
				help='Retain trees within this distance of uploaded water bodies. Upload water-body polygons and enter an EPSG code.\n\nRequires the `x_coord` and `y_coord` columns.',
			)
			st.number_input(
				"Road harvest buffer distance (m)", min_value=0.0, step=1.0,
				key="road_buffer_distance",
				disabled=not bool(st.session_state.get("roads_gpkg_bytes")),
				help="Harvest all trees within this distance of roads, including overlaps with water protection.\n\nRequires the `x_coord` and `y_coord` columns and a roads GeoPackage.",
			)
		with thinning_cols[2]:
			st.subheader("Resistance and resilience")
		with thinning_cols[3]:
			st.subheader("Silvicultural specifications", help="Beware of selecting too many constraints: the optimisation may become infeasible.")
			st.checkbox("Specify species distribution", key="specify_species_distribution", disabled=multi_stand or not bool(st.session_state.column_mapping.get("species")), help='Set the proportion of trees from each species to leave in the stand.\n\nNot compatible with multi-stand optimisation.\n\nRequires the `species` column.')
			st.checkbox("Specify DBH distribution", key="specify_dbh_distribution", disabled=multi_stand or dbh_values.empty, help='Set the proportion of trees from each DBH class to leave in the stand.\n\nNot compatible with multi-stand optimisation.\n\nRequires the `dbh` column.')

			st.checkbox(
				"Minimum distance between retained trees",
				key="constraint_min_distance",
				help=(
					'Set minimum spacing between retained trees, independently within each uploaded stand.\n\nRequires the `x_coord` and `y_coord` columns.'
				),
			)
			if st.session_state.constraint_min_distance:
				st.number_input(
					"Retained-tree minimum distance (m)",
					min_value=0.1,
					step=0.5,
					format="%.1f",
					key="constraint_min_distance_value",
				)

			st.checkbox(
				"Retained basal area", key="constraint_basal_area",
				disabled=multi_stand or dbh_values.empty,
				help='Set the target basal area left standing per hectare among buffer-eligible trees. Enter an EPSG code to calculate the area.\n\nNot compatible with multi-stand optimisation.\n\nRequires the `dbh`, `x_coord` and `y_coord` columns.',
			)
			if st.session_state.constraint_basal_area:
				total_basal_area = None
				epsg_text = st.session_state.epsg_text.strip()
				has_coordinates = bool(st.session_state.column_mapping.get("x_coord") and st.session_state.column_mapping.get("y_coord"))
				if not epsg_text.isdigit() or not has_coordinates:
					st.info("Map the X and Y coordinate columns and enter their EPSG code to calculate basal area per hectare.")
				else:
					try:
						preview_dataset = build_dataset(df, st.session_state.column_mapping, epsg_text)
						water_bytes = st.session_state.get("water_shapefile_bytes")
						road_bytes = st.session_state.get("roads_gpkg_bytes")
						apply_water_protection(preview_dataset, read_geometry_shapefile(water_bytes, preview_dataset.epsg) if water_bytes else None, st.session_state.water_buffer_distance)
						apply_road_harvest(preview_dataset, read_roads_geopackage(road_bytes) if road_bytes else None, st.session_state.road_buffer_distance)
						total_basal_area = total_basal_area_per_hectare(preview_dataset, eligible_only=True)
					except Exception as exc:
						st.warning(f"Basal area range is unavailable: {exc}")

				if total_basal_area is not None and total_basal_area > 0:
					keep_fraction = float(np.clip(
						st.session_state.constraint_basal_area_keep_fraction, 0.0, 1.0
					))
					weights = objective_weight_slider(
						objectives=["keep", "remove"],
						labels={"keep": "Basal area keep", "remove": "Basal area remove"},
						default_weights={"keep": keep_fraction, "remove": 1 - keep_fraction},
						key=f"basal_area_slider_{total_basal_area:.6f}",
						step=min(0.001, 0.01 / total_basal_area),
						colors=["#009E73", "#D55E00"],
						value_scale=total_basal_area, value_suffix=" m2/ha",
						show_percentage=True,
						show_equal_button=False,
						boundary_band=0.05,
						boundary_colors=["#007A59", "#A94400"],
					)
					st.session_state.constraint_basal_area_keep_fraction = weights["keep"]

		st.markdown("</div>", unsafe_allow_html=True)
		statistics_df = prepare_buffer_statistics(df)
		if st.session_state.specify_species_distribution or st.session_state.specify_dbh_distribution:
			render_frequency_specifications(statistics_df)
		if st.session_state.objective_min_volume:
			st.info("Only one objective is active, so its weight is fixed at 1.")
			return {"minvolume": 1.0}
		st.warning("No objectives selected. Map the volume column, then select at least one objective.")
		return {}
	wood_quality_required = bool(st.session_state.column_mapping.get("wood_quality"))
	species_required = bool(st.session_state.column_mapping.get("species"))
	alive_required = bool(st.session_state.column_mapping.get("is_alive"))

	objective_cols = st.columns(4)
	with objective_cols[0]:
		st.subheader("Economic return")
		st.checkbox(
			"Maximise social status",
			key="objective_social_status",
			disabled=not bool(st.session_state.column_mapping.get("social_status")),
			help='Prefer trees with a high social status relative to their neighbours.\n\nRequires the `social_status` column.',
		)
		st.checkbox(
			"Maximise wood quality",
			key="objective_wood_quality",
			disabled=not wood_quality_required,
			help='Prefer trees with high wood quality.\n\nRequires the `wood_quality` column.',
		)


	with objective_cols[1]:
		st.subheader("Biodiversity")
		st.info("No biodiversity objectives or constraints yet.")

	with objective_cols[2]:
		st.subheader("Resistance and resilience")
		st.info("No resistance & resilience objectives or constraints yet.")

	with objective_cols[3]:
		st.subheader("Silvicultural specifications", help="Beware of selecting too many constraints: the optimisation may become infeasible.")

		st.checkbox(
			"Enforce minimum distance between Z-trees",
			key="constraint_min_distance",
			help='Set minimum spacing between selected future crop trees.\n\nRequires the `x_coord` and `y_coord` columns.'
		)

		if st.session_state.constraint_min_distance:
			st.number_input(
				"Minimum distance (m)",
				min_value=0.1,
				step=0.5,
				format="%.1f",
				key="constraint_min_distance_value",
				help='Set minimum spacing between selected future crop trees.\n\nRequires the `x_coord` and `y_coord` columns.',
			)

		st.checkbox(
			"Space Z-trees on a grid",
			key="constraint_grid",
			help='Divide the area into a grid and limit future crop trees to one per cell.\n\nRequires the `x_coord` and `y_coord` columns.'
		)

		if st.session_state.constraint_grid:
			st.radio(
				"Grid type",
				options=["square", "hexagon"],
				format_func=lambda value: "Square" if value == "square" else "Hexagon",
				horizontal=True,
				key="constraint_grid_type",
			)
			st.number_input(
				"Grid size (m)",
				min_value=1,
				step=1,
				format="%d",
				key="constraint_grid_size",
			)

		st.checkbox(
			"Select only alive Z-trees",
			key="constraint_only_living_trees",
			disabled=not alive_required,
			help='Select only living future crop trees.\n\nRequires the `is_alive` column.',
		)

		st.checkbox("Specify species distribution", key="specify_species_distribution", disabled=multi_stand or not species_required, help='Set the proportion of trees from each species to leave in the stand.\n\nNot compatible with multi-stand optimisation.\n\nRequires the `species` column.')
		st.checkbox("Specify DBH distribution", key="specify_dbh_distribution", disabled=multi_stand or dbh_values.empty, help='Set the proportion of trees from each DBH class to leave in the stand.\n\nNot compatible with multi-stand optimisation.\n\nRequires the `dbh` column.')

		st.checkbox(
			"Density of Z trees per hectare",
			key="constraint_density",
			help='Set minimum and maximum numbers of future crop trees per hectare. Enter an EPSG code to calculate the area.\n\nRequires the `x_coord` and `y_coord` columns.',
		)
		
		if st.session_state.constraint_density:
			col1, col2 = st.columns(2)
			with col1:
				st.number_input(
					"Min density (trees/ha)",
					min_value=0,
					step=1,
					format="%d",
					key="constraint_density_min",
				)

			with col2:
				st.number_input(
					"Max density (trees/ha)",
					min_value=0,
					step=1,
					format="%d",
					key="constraint_density_max",
				)
			st.checkbox(
				"Enforce spatial spread of Z trees",
				key="constraint_spatial_spread",
				help='Spread future crop trees using the same minimum and maximum density bounds.\n\nRequires the `x_coord` and `y_coord` columns.',
			)

	if st.session_state.specify_species_distribution or st.session_state.specify_dbh_distribution:
		render_frequency_specifications(df)

	obj_weights = render_objective_weights()

	st.markdown("</div>", unsafe_allow_html=True)

	return obj_weights


def render_algorithm_section() -> None:
	st.markdown("<div class='step-title'>4. Algorithm choice</div>", unsafe_allow_html=True)


	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	render_algorithm_choices()

	st.markdown("---")

	if st.session_state.selected_algorithm == "linear_programming":
		st.selectbox(
			"Solver",
			options=AVAILABLE_LP_SOLVERS,
			format_func=lambda solver: (
				"CBC (included)" if solver == "CBC" else "Gurobi (licensed)"
			),
			help="CBC needs no licence. Gurobi uses your own Web License Service (WLS) credentials.",
			key="lp_solver",
			on_change=clear_gurobi_credentials,
		)
		st.selectbox(
				"Max runtime (seconds)",
				[30, 60, 120, 300, 600, 1200],
				key="lp_max_runtime",
		)
		if st.session_state.lp_solver == "GUROBI":
			st.markdown("##### Gurobi WLS licence")
			st.text_input("Access ID", type="password", key="gurobi_wls_access_id")
			st.text_input("Secret key", type="password", key="gurobi_wls_secret")
			st.text_input("Licence ID", key="gurobi_wls_license_id")
			st.caption(
				"Use a WLS API key valid for this cloud host. Credentials are sent to "
				"the hosting server and used only in your session; they are not saved "
				"to files or included in downloads. Desktop activation keys are not supported."
			)
			st.button("Clear Gurobi credentials", on_click=clear_gurobi_credentials)
	elif st.session_state.selected_algorithm == "differential_evolution":
		st.session_state.de_population_size = st.number_input(
			"Population size multiplier",
			min_value=1,
			max_value=100,
			value=int(st.session_state.de_population_size),
			help="SciPy creates this many candidates per decision variable.",
		)
		st.session_state.de_iterations = st.number_input(
			"Maximum iterations",
			min_value=1,
			max_value=10000,
			value=int(st.session_state.de_iterations),
		)
	elif st.session_state.selected_algorithm == "dual_annealing":
		st.session_state.da_iterations = st.number_input(
			"Maximum iterations",
			min_value=1,
			max_value=100000,
			value=int(st.session_state.da_iterations),
		)
		st.session_state.da_initial_temp = st.number_input(
			"Initial temperature",
			min_value=0.01,
			max_value=50000.0,
			value=float(st.session_state.da_initial_temp),
			help="Higher values permit broader exploration early in the search.",
		)
		st.session_state.da_restart_temp_ratio = st.number_input(
			"Restart temperature ratio",
			min_value=1e-12,
			max_value=0.999999,
			value=float(st.session_state.da_restart_temp_ratio),
			format="%.8f",
			help="Triggers re-annealing when the temperature falls below this fraction of the initial temperature.",
		)
		st.session_state.da_visit = st.number_input(
			"Visiting distribution parameter",
			min_value=1.000001,
			max_value=3.0,
			value=float(st.session_state.da_visit),
			help="Higher values produce longer jumps through the search space.",
		)
		st.session_state.da_accept = st.number_input(
			"Acceptance parameter",
			min_value=-10000.0,
			max_value=-5.0,
			value=float(st.session_state.da_accept),
			help="More negative values reduce the probability of accepting worse solutions.",
		)
		st.session_state.da_maxfun = st.number_input(
			"Maximum objective evaluations",
			min_value=1,
			max_value=1000000000,
			value=int(st.session_state.da_maxfun),
			help="Soft limit on calls to the objective function.",
		)
	elif st.session_state.selected_algorithm == "genetic_algorithm":
		st.session_state.ga_num_generations = st.number_input(
			"Number of generations",
			min_value=1,
			max_value=100000,
			value=int(st.session_state.ga_num_generations),
		)
		st.session_state.ga_sol_per_pop = st.number_input(
			"Solutions per population",
			min_value=2,
			max_value=10000,
			value=int(st.session_state.ga_sol_per_pop),
		)
		population_size = int(st.session_state.ga_sol_per_pop)
		st.session_state.ga_num_parents_mating = min(
			int(st.session_state.ga_num_parents_mating), population_size
		)
		st.session_state.ga_num_parents_mating = st.number_input(
			"Parents mating per generation",
			min_value=1,
			max_value=population_size,
			value=int(st.session_state.ga_num_parents_mating),
		)
		parent_selection_types = ["sss", "rws", "sus", "rank", "random", "tournament"]
		st.session_state.ga_parent_selection_type = st.selectbox(
			"Parent selection type",
			parent_selection_types,
			index=parent_selection_types.index(st.session_state.ga_parent_selection_type),
			help="Steady-state, roulette-wheel, stochastic-universal, rank, random, or tournament selection.",
		)
		st.session_state.ga_keep_elitism = min(
			int(st.session_state.ga_keep_elitism), population_size
		)
		st.session_state.ga_keep_elitism = st.number_input(
			"Elite solutions retained",
			min_value=0,
			max_value=population_size,
			value=int(st.session_state.ga_keep_elitism),
		)
		crossover_types = ["single_point", "two_points", "uniform", "scattered"]
		st.session_state.ga_crossover_type = st.selectbox(
			"Crossover type",
			crossover_types,
			index=crossover_types.index(st.session_state.ga_crossover_type),
		)
		mutation_types = ["random", "swap", "inversion", "scramble"]
		st.session_state.ga_mutation_type = st.selectbox(
			"Mutation type",
			mutation_types,
			index=mutation_types.index(st.session_state.ga_mutation_type),
		)
		st.session_state.ga_mutation_probability = st.number_input(
			"Mutation probability per gene",
			min_value=0.0,
			max_value=1.0,
			value=float(st.session_state.ga_mutation_probability),
			step=0.01,
		)
		st.session_state.ga_use_random_seed = st.checkbox(
			"Use a fixed random seed",
			value=bool(st.session_state.ga_use_random_seed),
		)
		if st.session_state.ga_use_random_seed:
			st.session_state.ga_random_seed = st.number_input(
				"Random seed",
				min_value=0,
				max_value=2147483647,
				value=int(st.session_state.ga_random_seed),
			)

	st.markdown("</div>", unsafe_allow_html=True)

def main() -> None:
	initialize_state()
	st.markdown(CSS, unsafe_allow_html=True)

	st.markdown(
		f"""
		<div class="hero">
			<h1>SingleTreeOpt</h1>
			<img class="hero-logo" src="{image_to_data_uri(IMAGE_DIR / 'ST PRINCIPAL negative.png')}" alt="SingleTree logo" />
		</div>
		""",
		unsafe_allow_html=True,
	)

	df = render_upload_section()
	render_problem_section()

	obj_weights = render_objectives_constraints_section(df)
	render_algorithm_section()

	st.markdown("<div class='step-card'>", unsafe_allow_html=True)
	st.checkbox(
		"Report objective scores as percentages",
		key="report_objectives_as_percentages",
		help="Show each objective score as a percentage of its constrained minimum-to-maximum range.",
	)
	run_clicked = st.button(
		"Run optimisation",
		key="run_optimisation",
		type="primary",
		use_container_width=True,
		disabled=False,
	)
	st.markdown("</div>", unsafe_allow_html=True)

	if run_clicked:
		if uses_gurobi_wls():
			try:
				session_wls_credentials()
			except GurobiLicenseError as exc:
				st.error(str(exc))
				return
		if df is None:
			st.error("Please select a case study dataset first.")
			return

		missing_roles = validation_report(st.session_state.column_mapping)
		if missing_roles:
			st.error("Map the following core columns before running: " + ", ".join(missing_roles))
		elif not any([
			st.session_state.objective_social_status,
			st.session_state.objective_wood_quality,
			st.session_state.objective_min_dbh,
			st.session_state.objective_min_volume,
		]):
			st.error("Select at least one objective before running the optimiser.")
		else:
			try:
				dataset = build_dataset(df, st.session_state.column_mapping, st.session_state.epsg_text)
			except Exception as exc:
				st.error(f"Unable to prepare the dataset: {exc}")
				st.code(traceback.format_exc())
				return

			stands = None
			try:
				epsg_code = int(st.session_state.epsg_text)
				stand_bytes = st.session_state.get("stand_shapefile_bytes")
				if stand_bytes:
					stands = read_geometry_shapefile(stand_bytes, epsg_code)
					assignment = assign_trees_to_stands(
						dataset, stands, int(st.session_state.stand_assignment_seed)
					)
					stands = assignment["stands"]
					if assignment["unassigned_trees"]:
						st.info(
							f"{assignment['unassigned_trees']:,} trees fall outside all stands. "
							"They are excluded from stand optimisations; road-buffer trees will still be harvested and all other unassigned trees retained."
						)
					st.session_state.spatial_validation = {
						"stand_count": len(stands),
						"boundary_ties": assignment["boundary_ties"],
						"unassigned_trees": assignment["unassigned_trees"],
					}
				else:
					dataset.data["CG_source_row"] = np.arange(dataset._n_trees, dtype=int)

				water = None
				water_bytes = st.session_state.get("water_shapefile_bytes")
				if water_bytes:
					water = read_geometry_shapefile(water_bytes, epsg_code)
				protected_count = apply_water_protection(
					dataset, water, st.session_state.water_buffer_distance
				)
				road_bytes = st.session_state.get("roads_gpkg_bytes") if is_thinning_treatment() else None
				roads = read_roads_geopackage(road_bytes) if road_bytes else None
				apply_road_harvest(dataset, roads, st.session_state.road_buffer_distance)
				protected_count = int(dataset.data[WaterProtectionConstraint.indicator_col].sum())
				dataset.buffer_inputs = {
					"roads_gpkg_bytes": road_bytes,
					"road_buffer_distance": st.session_state.road_buffer_distance,
					"water_shapefile_bytes": water_bytes,
					"water_buffer_distance": st.session_state.water_buffer_distance,
				}
			except Exception as exc:
				st.error(f"Unable to prepare spatial treatment inputs: {exc}")
				st.code(traceback.format_exc())
				return

			if stands is not None:
				try:
					results, stand_results = run_with_solver_license(
						optimise_stands, dataset, stands, obj_weights
					)
				except GurobiLicenseError as exc:
					st.error(str(exc))
					return
				st.session_state.optimisation_results = results
				st.session_state.optimisation_algorithm = st.session_state.selected_algorithm
				st.session_state.optimisation_dataset = dataset
				st.session_state.optimisation_objectives = {}
				st.session_state.optimisation_constraints = []
				st.session_state.optimisation_benchmarks = None
				st.session_state.optimisation_optimiser = None
				st.session_state.optimisation_stand_results = stand_results
				st.session_state.optimisation_report_percentages = False
				st.session_state.desired_species_frequency = None
				st.session_state.desired_dbh_frequency = None
				st.session_state.desired_species_labels = []
				st.session_state.desired_dbh_edges = []
				st.session_state.thinning_density_targets = []
				st.session_state.thinning_schedule = None
				st.success(
					f"Completed {len(stand_results):,} stand optimisations; "
					f"{protected_count:,} trees are protected by the water buffer."
				)
				st.rerun()

			objectives = build_objectives(dataset, obj_weights)
			constraints = build_constraints(dataset)
			optimiser = build_optimiser(dataset, objectives, constraints)
			if st.session_state.selected_algorithm == "differential_evolution":
				de_progress_text = st.empty()
				de_progress_text.write("Evaluating initial DE population")
				de_generation = {"value": 0}
				de_max_iterations = int(st.session_state.de_iterations)

				def update_de_progress(intermediate_result):
					de_generation["value"] += 1
					de_progress_text.write(
						f"Running DE - generation {de_generation['value']} of {de_max_iterations}"
					)

				algorithm_hyperparameters = {
					"maxiter": de_max_iterations,
					"popsize": int(st.session_state.de_population_size),
					"progress_callback": update_de_progress,
				}
			elif st.session_state.selected_algorithm == "linear_programming":
				algorithm_hyperparameters = lp_hyperparameters()
			elif st.session_state.selected_algorithm == "dual_annealing":
				algorithm_hyperparameters = {
					"maxiter": int(st.session_state.da_iterations),
					"initial_temp": float(st.session_state.da_initial_temp),
					"restart_temp_ratio": float(st.session_state.da_restart_temp_ratio),
					"visit": float(st.session_state.da_visit),
					"accept": float(st.session_state.da_accept),
					"maxfun": int(st.session_state.da_maxfun),
				}
			elif st.session_state.selected_algorithm == "genetic_algorithm":
				ga_progress_text = st.empty()
				ga_progress_text.write("Evaluating initial GA population")
				ga_max_generations = int(st.session_state.ga_num_generations)

				def update_ga_progress(ga_instance):
					ga_progress_text.write(
						"Running GA - generation "
						f"{ga_instance.generations_completed} of {ga_max_generations}"
					)

				algorithm_hyperparameters = {
					"num_generations": ga_max_generations,
					"sol_per_pop": int(st.session_state.ga_sol_per_pop),
					"num_parents_mating": int(st.session_state.ga_num_parents_mating),
					"parent_selection_type": st.session_state.ga_parent_selection_type,
					"keep_elitism": int(st.session_state.ga_keep_elitism),
					"crossover_type": st.session_state.ga_crossover_type,
					"mutation_type": st.session_state.ga_mutation_type,
					"mutation_probability": float(st.session_state.ga_mutation_probability),
					"random_seed": (
						int(st.session_state.ga_random_seed)
						if st.session_state.ga_use_random_seed else None
					),
					"progress_callback": update_ga_progress,
				}
			else:
				raise ValueError(
					f"Unsupported algorithm: {st.session_state.selected_algorithm}"
				)
			try:
				results = run_with_solver_license(
					optimiser.optimise, algorithm_hyperparameters=algorithm_hyperparameters
				)
			except GurobiLicenseError as exc:
				st.error(str(exc))
				return
			if is_thinning_treatment():
				try:
					validate_buffer_decisions(dataset, results)
				except ValueError as exc:
					st.error(str(exc))
					return
			benchmarks = None
			if st.session_state.report_objectives_as_percentages:
				try:
					benchmarks = run_with_solver_license(
						optimiser.benchmark_objectives, results["decisions"],
						algorithm_hyperparameters=lp_hyperparameters(),
					)
				except GurobiLicenseError as exc:
					st.warning(str(exc))
			st.session_state.optimisation_results = results
			st.session_state.optimisation_algorithm = st.session_state.selected_algorithm
			st.session_state.optimisation_dataset = dataset
			st.session_state.optimisation_objectives = objectives
			st.session_state.optimisation_constraints = constraints
			st.session_state.optimisation_benchmarks = benchmarks
			st.session_state.optimisation_optimiser = optimiser
			st.session_state.optimisation_stand_results = None
			st.session_state.desired_species_frequency = None
			st.session_state.desired_dbh_frequency = None
			st.session_state.desired_species_labels = []
			st.session_state.desired_dbh_edges = []
			st.session_state.thinning_density_targets = []
			st.session_state.thinning_schedule = None
			st.session_state.optimisation_report_percentages = bool(
				st.session_state.report_objectives_as_percentages
			)
			st.success("Optimisation completed.")

	if st.session_state.get("optimisation_results"):
		render_optimisation_results()
		render_results_download(
			st.session_state.optimisation_dataset,
			st.session_state.optimisation_results,
		)


if __name__ == "__main__":
	main()
