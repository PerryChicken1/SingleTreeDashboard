"""Run from an exported dashboard, including in GitHub's Linux CI."""
from io import BytesIO
import json
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from streamlit.testing.v1 import AppTest
from streamlit.proto.TextInput_pb2 import TextInput

ROOT = Path(__file__).resolve().parent


def main():
    # All imports must resolve inside this small release, without the source repo.
    import Dashboard as dashboard
    from Constraints import ProportionConstraint
    from Objectives import MinDBHObjective, MinVolumeObjective
    from Optimiser import FutureCropTreeOptimiser, ThinningOptimiser
    from SingleTreeAlgorithms import LinearProgrammingAlgorithm

    metadata = json.loads((ROOT / "app/static/metadata.json").read_text(encoding="utf-8"))
    images = ROOT / "app/static" / metadata["image_dir"]
    for section in ("problems", "algorithms"):
        for item in metadata[section].values():
            assert (images / item["image"]).is_file(), item["image"]
    assert (images / "ST PRINCIPAL negative.png").is_file()
    from PIL import Image
    for preset in dashboard.TEST_DATASETS.values():
        with Image.open(images / preset["image"]) as image:
            assert image.size == (1200, 750)
    for name in ("objective_weight_slider", "thinning_schedule_chart", "frequency_histogram"):
        assert (ROOT / "app/components" / name / "frontend/build/index.html").is_file()

    frame = pd.DataFrame({
        "id": ["c", "a", "b"], "x": [400000., 400020., 400000.],
        "y": [6780000., 6780000., 6780020.], "dbh": [10., 20., 30.],
        "volume": [1., 2., 3.], "species": ["pine"] * 3,
    }, index=[9, 4, 7])
    mapping = {"id": "id", "x_coord": "x", "y_coord": "y", "dbh": "dbh", "volume": "volume"}
    for optimiser_class in (FutureCropTreeOptimiser, ThinningOptimiser):
        dataset = dashboard.build_dataset(frame, mapping, "3067")
        # Two objectives exercise auxiliary normalization solves, whose default
        # solver must also work without Gurobi.
        optimiser = optimiser_class(
            dataset, {MinDBHObjective(): 0.5, MinVolumeObjective(): 0.5},
            [ProportionConstraint(lb=1/3, ub=1/3, prop_col="species", prop_vals={"pine"})],
            LinearProgrammingAlgorithm(),
        )
        result = optimiser.optimise({"solver_name": "CBC", "max_time": 10, "num_workers": 1})
        assert result["is_optimal"], result
        assert result["decision_vector"].tolist() == [1, 0, 0], result
        output = pd.read_csv(BytesIO(dashboard.results_csv(dataset, result)))
        assert output.columns.tolist() == list(frame.columns) + ["optimisation_decision"]
        assert output["id"].tolist() == frame["id"].tolist()
        assert output["optimisation_decision"].tolist() == [1, 0, 0]

    # GDAL/PROJ must work through the installed wheels on the hosting OS.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roads.gpkg"
        roads = gpd.GeoDataFrame(geometry=[LineString([(400000, 6780000), (400020, 6780020)])], crs=3067)
        roads.to_file(path, layer="skid_roads", driver="GPKG")
        loaded = dashboard.read_roads_geopackage(path.read_bytes())
        assert len(loaded) == 1 and loaded.crs.to_epsg() == 3067

    app = AppTest.from_file(str(ROOT / "Dashboard.py"), default_timeout=60).run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state["lp_solver"] in dashboard.AVAILABLE_LP_SOLVERS
    assert app.session_state["lp_solver"] == "CBC"
    assert not app.get("file_uploader")
    assert not any(item.key == "epsg_text" for item in app.text_input)
    app.selectbox(key="lp_solver").select("GUROBI").run()
    assert not app.exception
    for key in ("gurobi_wls_access_id", "gurobi_wls_secret"):
        assert app.text_input(key=key).proto.type == TextInput.PASSWORD
    app.button(key="run_optimisation").click().run()
    assert any("WLS access ID" in item.value for item in app.error)
    app.text_input(key="gurobi_wls_secret").set_value("test-secret").run()
    next(button for button in app.button if button.label == "Clear Gurobi credentials").click().run()
    assert app.session_state["gurobi_wls_secret"] == ""
    other_user = AppTest.from_file(str(ROOT / "Dashboard.py"), default_timeout=60).run()
    assert other_user.session_state["gurobi_wls_secret"] == ""
    app.selectbox(key="lp_solver").select("CBC").run()
    assert not app.exception
    app.button(key="example_basel").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state["epsg_text"] == "2056"
    assert app.session_state["selected_problem"] == "future_crop_tree_selection"
    assert all(value is None for value in app.session_state["column_mapping"].values())
    assert all(item.value == "Not set" for item in app.selectbox if item.key and item.key.endswith("_column"))
    assert not any(item.value.startswith("Selected dataset:") for item in app.caption)
    assert app.button(key="problem_thinning_treatment").disabled
    assert not app.button(key="problem_future_crop_tree_selection").disabled
    assert "problem_risk_mitigation" not in {item.key for item in app.button}
    assert not app.get("file_uploader")
    assert not any(item.key == "epsg_text" for item in app.text_input)
    for key, column in {
        "id_column": "Tree ID", "x_column": "X-coord. [m]", "y_column": "Y-coord. [m]",
        "dbh_column": "DBH [cm]", "social_status_column": "Social status", "is_alive_column": "Alive",
    }.items():
        app.selectbox(key=key).select(column)
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state["column_mapping"]["dbh"] == "DBH [cm]"
    app.run()
    assert app.session_state["column_mapping"]["dbh"] == "DBH [cm]"
    assert any(item.value == "Data preview" for item in app.caption)
    assert any("Planned attributes (WP1)" in item.value for item in app.markdown)
    assert not any(item.label in {"Minimise DBH", "Map preview and polygon filtering"} for item in app.checkbox)
    algorithm_keys = {item.key for item in app.button if item.key and item.key.startswith("algorithm_")}
    assert algorithm_keys == {"algorithm_linear_programming"}
    app.checkbox(key="objective_social_status").check().run()
    app.button(key="run_optimisation").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state["optimisation_results"]["treatment_type"] == "future_crop_tree_selection"
    app.button(key="example_evo").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state["epsg_text"] == "3067"
    assert app.session_state["selected_problem"] == "thinning_treatment"
    assert all(value is None for value in app.session_state["column_mapping"].values())
    assert all(item.value == "Not set" for item in app.selectbox if item.key and item.key.endswith("_column"))
    assert not any(item.value.startswith("Selected dataset:") for item in app.caption)
    assert app.button(key="problem_future_crop_tree_selection").disabled
    assert not app.button(key="problem_thinning_treatment").disabled
    assert not app.get("file_uploader")
    assert app.session_state["water_shapefile_bytes"]
    assert app.session_state["roads_gpkg_bytes"]
    assert app.session_state["stand_shapefile_bytes"] is None
    preset = dashboard.TEST_DATASETS["evo"]
    bundled_water = ROOT / "data/examples" / preset["water_file"]
    evo_trees = pd.read_csv(ROOT / "data/examples" / preset["file"])
    stand = gpd.read_file(ROOT / "data/examples/evo_stand_050/stand.gpkg")
    assert len(evo_trees) == 301
    points = gpd.GeoSeries(gpd.points_from_xy(evo_trees.LX, evo_trees.LY), crs=3067)
    assert points.intersects(stand.geometry.iloc[0]).all()
    water = gpd.read_file(bundled_water)
    assert len(water) == 1 and water["id"].iloc[0] == 25837698
    roads = dashboard.read_roads_geopackage(app.session_state["roads_gpkg_bytes"])
    assert len(roads) == 1 and roads.trail_id.iloc[0] == 9
    assert roads.intersects(stand.geometry.iloc[0]).all()
    assert gpd.read_file(bundled_water).crs.to_epsg() == 3067
    assert dashboard.read_roads_geopackage(app.session_state["roads_gpkg_bytes"]).crs.to_epsg() == 3067
    assert app.session_state["optimisation_results"] is None
    app.selectbox(key="id_column").select("ID").run()
    for key, column in {"x_column": "LX", "y_column": "LY", "dbh_column": "DBH",
                        "volume_column": "Volume"}.items():
        app.selectbox(key=key).select(column)
    app.run()
    app.checkbox(key="objective_min_volume").check().run()
    app.button(key="run_optimisation").click().run()
    assert not app.exception, [item.message for item in app.exception]
    app.checkbox(key="show_decision_map").check().run()
    assert not app.exception, [item.message for item in app.exception]
    for key in ("show_water_bodies_layer", "show_roads_layer"):
        assert app.checkbox(key=key).value is True
        app.checkbox(key=key).uncheck().run()
        assert app.checkbox(key=key).value is False
    app.checkbox(key="show_decision_map").uncheck().run()
    app.checkbox(key="show_decision_map").check().run()
    assert app.checkbox(key="show_water_bodies_layer").value is True
    assert app.checkbox(key="show_roads_layer").value is True
    app.button(key="example_basel").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert all(value is None for value in app.session_state["column_mapping"].values())
    assert all(item.value == "Not set" for item in app.selectbox if item.key and item.key.endswith("_column"))
    assert not any(item.value.startswith("Selected dataset:") for item in app.caption)
    assert app.session_state["water_shapefile_bytes"] is None
    assert app.session_state["roads_gpkg_bytes"] is None
    assert app.session_state["selected_problem"] == "future_crop_tree_selection"
    app.selectbox(key="id_column").select("Tree ID").run()
    app.button(key="example_basel").click().run()
    assert not app.exception
    assert app.selectbox(key="id_column").value == "Not set"
    for labels in (("Not selected", "Selected"), ("Retain", "Cut")):
        for figure in (
            dashboard.decision_histogram(pd.Series([10., 20.]), np.array([False, True]), np.array([0., 15., 30.]), "DBH", labels),
            dashboard.species_decision_histogram(pd.Series(["pine", "spruce"]), np.array([False, True]), labels),
        ):
            assert [trace.name for trace in figure.data] == list(labels)
            assert [trace.marker.color for trace in figure.data] == ["#009E73", "#D55E00"]
    assert not (ROOT / "app/decisions.pkl").exists()
    print("PASS: standalone startup, assets, CBC multi-objective crop/thinning solves, CSV export, GeoPackage/CRS.")


if __name__ == "__main__":
    main()
