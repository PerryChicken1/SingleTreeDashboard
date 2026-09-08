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
    assert not (ROOT / "app/decisions.pkl").exists()
    print("PASS: standalone startup, assets, CBC multi-objective crop/thinning solves, CSV export, GeoPackage/CRS.")


if __name__ == "__main__":
    main()
