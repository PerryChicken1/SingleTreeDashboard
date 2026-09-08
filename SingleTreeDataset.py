import pandas as pd
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from shapely.geometry import shape
from DecisionScope import DecisionScope, coerce_decision_scope

class SingleTreeDataset:
    """A dataset of individual trees."""

    column_dtypes = {
        'id': 'string',
        'dbh': 'float64',
        'x': 'float64',
        'y': 'float64',
        'species': 'string',
        'social_status': 'float64',
        'is_alive': 'boolean',
        'pruned': 'boolean',
        'quality': 'float64',
        'volume': 'float64',
    }

    def __init__(self, data: pd.DataFrame):
        
        # a DataFrame with trees as rows and their attributes as columns
        self.data = data

        self.columns = self.init_columns()
        self.hectares = None

    ##################
    # INITIALISATION #
    ##################

    @staticmethod
    def init_columns() -> dict:
        """Initialize dataset column names."""

        return {
            'id': None,
            'dbh': None,
            'x': None,
            'y': None,
            'species': None,
            'social_status': None,
            'is_alive': None,
            'pruned': None,
            'quality': None,
            'volume': None
            }
    
    def supported_columns(self):
        """Print the set of supported column names."""
        print(set(self.columns.keys()))

    
    ###########
    # GETTERS #
    ###########

    @property
    def epsg(self):
        return self._epsg

    @property
    def _n_trees(self):
        return len(self.data)
    
    def _get_column_name(self, column_name: str):
        """Get name in dataset of standardised column."""
        assert column_name in self.columns.keys(), f"Column name {column_name} not recognised. Must be one of {list(self.columns.keys())}"
        standard_name = self.columns[column_name]
        assert standard_name is not None, f"Column name for '{column_name}' must be set... try `dataset.set_columns()`"
        return standard_name
    
    def validate_decision_vector(self, decisions) -> np.ndarray:
        """Validate and return a positional binary decision vector."""
        vector = np.asarray(decisions)
        if vector.ndim != 1 or vector.size != self._n_trees:
            raise ValueError(
                "Decision vector must be one-dimensional and contain one entry "
                f"per tree ({self._n_trees})."
            )
        if not np.all(np.isin(vector, (0, 1))):
            raise ValueError("Decision vector entries must all be 0 or 1.")
        return vector.astype(np.int8, copy=False)

    def decision_mask(self, decisions, applies_to: DecisionScope | str) -> np.ndarray:
        """Return the positional boolean mask for one side of a decision."""
        vector = self.validate_decision_vector(decisions)
        scope = coerce_decision_scope(applies_to)
        return vector.astype(bool) if scope is DecisionScope.SELECT else ~vector.astype(bool)

    def decision_indices(self, decisions, applies_to: DecisionScope | str) -> list[int]:
        """Return positional indices for selected or not-selected trees."""
        return np.flatnonzero(self.decision_mask(decisions, applies_to)).tolist()

    def get_decision_trees(self, decisions, applies_to: DecisionScope | str) -> pd.DataFrame:
        """Return trees in a decision scope, preserving row-position semantics."""
        return self.data.iloc[self.decision_mask(decisions, applies_to)]

    def _get_selected_trees(self, decisions) -> pd.DataFrame:
        """Backward-compatible alias for the ``SELECT`` side of a vector."""
        return self.get_decision_trees(decisions, DecisionScope.SELECT)
    
    ###########
    # SETTERS #
    ###########

    def set_columns(self, **kwargs) -> None:
        """Specify dataset column names."""
        for k, v in kwargs.items():
            assert k in self.columns.keys(), f"Column name {k} not recognised. Must be one of {list(self.columns.keys())}"
            assert v in self.data.columns, f"Column name {v} not found in dataset."
            self.columns[k] = v

    def convert_mapped_column_types(self) -> None:
        """Convert every mapped column to its canonical internal dtype."""
        for role, column_name in self.columns.items():
            if column_name is None:
                continue
            target_type = self.column_dtypes[role]
            try:
                self.data[column_name] = self.data[column_name].astype(target_type)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to convert column '{column_name}' to type {target_type}."
                ) from exc

    @epsg.setter
    def epsg(self, value: int) -> None:
        """Set EPSG code for dataset."""
        self._epsg = value
        self.geodatafy()
        self.calculate_area()

    ############
    # PLOTTERS #
    ############

    def geodatafy(self):
        """Convert dataset to GeoDataFrame."""

        # if already GeoDataFrame, do nothing
        if isinstance(self.data, gpd.GeoDataFrame):
            return
        
        assert self.epsg is not None, "EPSG code must be set... try dataset.epsg = <EPSG code>"
        assert self.columns['x'] is not None and self.columns['y'] is not None, "X and Y column names must be set... try `dataset.set_columns()`"

        gdf = gpd.GeoDataFrame(
                self.data,
                geometry=gpd.points_from_xy(self.data[self.columns['x']], self.data[self.columns['y']]),
                crs=f"EPSG:{self.epsg}"
            )

        self.data = gdf

    def folium_plot(self, save: bool = False):
        """Plot tree locations using folium."""
        
        # TODO: add option to specify which column is used for colour-coding the points

        data_wgs84 = self.data.to_crs("EPSG:4326")
        
        m = folium.Map(location=[data_wgs84.geometry.y.mean()
                                 , data_wgs84.geometry.x.mean()]
                                 , zoom_start=12)

        folium.TileLayer('OpenStreetMap').add_to(m)

        for _, row in data_wgs84.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green'
            ).add_to(m)

        if save:
            output_path = Path(__file__).resolve().parent / "results" / "maps" / "tree_locations.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.save(output_path)

        return m

    @staticmethod
    def filter_from_polygon(dataset: "SingleTreeDataset", polygon):
        """Return a copy of a geospatial tree dataset filtered to points inside a polygon."""
        assert isinstance(dataset.data, gpd.GeoDataFrame), "dataset.data must be a GeoDataFrame."

        if isinstance(polygon, dict):
            if polygon.get("type") == "Feature":
                polygon = polygon.get("geometry")
            polygon_geometry = shape(polygon)
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_geometry], crs="EPSG:4326")
            if dataset.data.crs is not None:
                polygon_geometry = polygon_gdf.to_crs(dataset.data.crs).geometry.iloc[0]
        else:
            polygon_geometry = polygon

        mask = dataset.data.geometry.intersects(polygon_geometry)
        filtered_dataset = SingleTreeDataset(dataset.data.loc[mask].copy().reset_index(drop=True))
        filtered_dataset.columns = dataset.columns.copy()
        filtered_dataset.hectares = dataset.hectares
        if hasattr(dataset, "_epsg"):
            filtered_dataset._epsg = dataset.epsg
        return filtered_dataset

    ############################
    # CALCULATE SPATIAL EXTENT #
    ############################

    @staticmethod
    def metric_conversion(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Project GeoDataFrame to metric coordinates."""
        utm_crs = gdf.estimate_utm_crs()
        gdf_metric = gdf.to_crs(utm_crs)
        return gdf_metric

    def calculate_area(self):
        """Calculate the area of the convex hull enclosing the tree points, in ha (= 10000 m²)."""
        gdf_metric = self.metric_conversion(self.data)
        hull = gdf_metric.union_all().convex_hull
        self.hectares = hull.area / 10000

    def calculate_hegyi_indices(self, Z_tree_indices):
        """Calculate each tree's Hegyi index relative to its nearest Z-tree.

        Distances are measured in metres. Z-trees themselves receive ``NaN``
        because they are reference trees rather than thinning candidates.
        """
        hegyi_indices, _ = self.calculate_hegyi_indices_with_nearest_z_tree(
            Z_tree_indices
        )
        return hegyi_indices

    def calculate_hegyi_indices_with_nearest_z_tree(self, Z_tree_indices):
        """Return Hegyi indices and the nearest Z-tree index for each tree."""
        Z_tree_indices = np.unique(Z_tree_indices)

        dbh_col = self._get_column_name('dbh')
        dbh = self.data[dbh_col].to_numpy(dtype=float, copy=False)
        metric_data = self.metric_conversion(self.data)
        coordinates = np.column_stack((
            metric_data.geometry.x.to_numpy(),
            metric_data.geometry.y.to_numpy(),
        ))
        distances, nearest_positions = cKDTree(
            coordinates[Z_tree_indices]
        ).query(coordinates, k=1)
        nearest_Z_tree_indices = Z_tree_indices[nearest_positions]

        Z_tree_mask = np.zeros(self._n_trees, dtype=bool)
        Z_tree_mask[Z_tree_indices] = True

        hegyi_indices = np.full(self._n_trees, np.nan, dtype=float)
        candidates = ~Z_tree_mask
        hegyi_indices[candidates] = (
            dbh[candidates]
            / (dbh[nearest_Z_tree_indices[candidates]] * distances[candidates])
        )
        return hegyi_indices, nearest_Z_tree_indices

