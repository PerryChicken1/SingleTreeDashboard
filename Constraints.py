from typing import Literal
from SingleTreeDataset import SingleTreeDataset
from abc import ABC, abstractmethod
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import cKDTree
from DecisionScope import DecisionScope, coerce_decision_scope

class Constraint(ABC):
    """Base class for constraints."""
    
    def __init__(self, name: str,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        
        self.name = name
        self.dataset = None
        self.applies_to = applies_to

    @property
    def applies_to(self) -> DecisionScope:
        return self._applies_to

    @applies_to.setter
    def applies_to(self, value: DecisionScope | str) -> None:
        self._applies_to = coerce_decision_scope(value)

    def bind(self, dataset: SingleTreeDataset) -> "Constraint":
        """Bind this reusable constraint definition to an inventory dataset."""
        self.dataset = dataset
        return self

    def decision_mask(self, decisions: list) -> np.ndarray:
        """Return the positional mask this constraint is defined over."""
        return self.dataset.decision_mask(decisions, self.applies_to)

    ###############
    # CALCULATION #
    ###############

    @abstractmethod
    def evaluate(self, decisions: list) -> float:
        """Evaluate constraint for decisions."""
        pass

    @abstractmethod
    def is_satisfied(self, decisions: list) -> bool:
        """Check if constraint is satisfied for decisions."""
        pass

    ###############
    # DESCRIPTION #
    ###############

    @abstractmethod
    def description(self):
        """Description of the constraint."""
        pass

    def describe(self):
        """Print description of the constraint."""
        print(f"Constraint: {self.name}\n\n")
        print(f"Description: {self.description()}")

    ##############
    # PROPERTIES #
    ##############

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, value):
        if value not in {'proportion', 'group_balance', 'count', 'pairwise_distance'}:
            raise ValueError(f"Constraint type '{value}' is not supported.")
        self._type = value

        
    ########################################
    # CONVERT FROM TREES/HA TO PROPORTION  #
    ########################################

    @staticmethod
    def density_to_proportion(trees_per_ha: float, hectares: float, total_trees: int) -> float:
        """Convert density in trees/ha to proportion of total trees."""
        proportion = (trees_per_ha * hectares) / total_trees
        return proportion


#########################################################
#                PROPORTION CONSTRAINTS                 #
#########################################################

class ProportionConstraint(Constraint):
    """Constraints on the proportion of trees with a particular property."""

    def __init__(self, name: str = "Proportion constraint",
                 lb: float = 0.0, ub: float = 1.0, prop_col: str = None,
                 prop_vals: set | None = None,
                 applies_to: DecisionScope | str = DecisionScope.SELECT):
        
        super().__init__(name=name, applies_to=applies_to)
        self.lb = lb
        self.ub = ub
        self.type = 'proportion'
        self.prop_col = prop_col
        self.prop_vals = set() if prop_vals is None else set(prop_vals)

    ###############
    # CALCULATION #
    ###############

    def evaluate(self, decisions: list) -> float:
        """Evaluate the matching-tree share of the full inventory."""
        mask = self.decision_mask(decisions)
        return float(self.linear_coefficients()[mask].sum())

    def linear_coefficients(self) -> np.ndarray:
        """Return coefficients for the constraint's single linear row."""
        return (
            self.dataset.data[self.prop_col]
            .isin(self.prop_vals)
            .to_numpy(dtype=float, copy=False)
            / self.dataset._n_trees
        )

    def is_satisfied(self, decisions: list) -> bool:
        """Constraint is satisfied if proportion of decisions is within bounds."""
        proportion = self.evaluate(decisions)
        return proportion >= self.lb and proportion <= self.ub

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        
        return f"""
        Proportion of matching decisions must be between {self.lb} and {self.ub}.
        """

class ZTreeProportionConstraint(ProportionConstraint):
    """Density of trees to mark as Z trees."""
    
    def __init__(self, lb_trees_per_ha: float = 10,
        ub_trees_per_ha: float = 20,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        self.lb_trees_per_ha = lb_trees_per_ha
        self.ub_trees_per_ha = ub_trees_per_ha
        super().__init__(name="Z tree proportion", lb=0.0, ub=1.0,
            prop_col='CG_redundant_indicator', prop_vals={1}, applies_to=applies_to)

    def bind(self, dataset: SingleTreeDataset) -> "ZTreeProportionConstraint":
        dataset = self.pad_ones(dataset)
        self.lb = self.density_to_proportion(
            self.lb_trees_per_ha, dataset.hectares, dataset._n_trees
        )
        self.ub = self.density_to_proportion(
            self.ub_trees_per_ha, dataset.hectares, dataset._n_trees
        )
        return super().bind(dataset)

    ##############
    # NEW COLUMN #
    ##############

    @staticmethod
    def pad_ones(dataset: SingleTreeDataset) -> SingleTreeDataset:
        """Add a new column to the dataset."""
        dataset.data['CG_redundant_indicator'] = 1
        return dataset

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        
        return f"""
        Density of Z trees must be between {self.lb} and {self.ub} of total trees.
        """
    
class DeadTreeConstraint(ProportionConstraint):
    """Limit selection of dead trees."""
    
    def __init__(self, lb: float = 0, ub: float = 0,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(name="Dead tree proportion", lb=lb, ub=ub,
            prop_col=None, prop_vals={0}, applies_to=applies_to)

    def bind(self, dataset: SingleTreeDataset) -> "DeadTreeConstraint":
        self.prop_col = dataset._get_column_name('is_alive')
        return super().bind(dataset)

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        
        return f"""
        Proportion of dead trees must be between {self.lb} and {self.ub} of total trees.
        """


class BasalAreaConstraint(ProportionConstraint):
    """Retain a target basal area in m2/ha; input DBH must be in centimetres."""

    basal_area_col = 'CG_basal_area'

    def __init__(self, basal_area_target: float = 0.0, margin: float = 0.0, eligible_col: str | None = None):

        self.eligible_col = eligible_col
        self.basal_area_target = basal_area_target
        self.margin = margin
        super().__init__(name="Retained basal area",
            lb=basal_area_target - margin / 2,
            ub=basal_area_target + margin / 2,
            prop_col=self.basal_area_col,
            applies_to=DecisionScope.NOT_SELECT)

    def bind(self, dataset: SingleTreeDataset) -> "BasalAreaConstraint":
        dbh_col = dataset._get_column_name('dbh')
        diameter_metres = dataset.data[dbh_col].to_numpy(dtype=float) / 100
        dataset.data[self.basal_area_col] = np.pi * (diameter_metres / 2) ** 2
        return super().bind(dataset)

    def linear_coefficients(self) -> np.ndarray:
        """Return eligible trees' basal-area contributions in m2/ha."""
        coefficients = self.dataset.data[self.basal_area_col].to_numpy(dtype=float, copy=False) / self.dataset.hectares
        if self.eligible_col is not None:
            coefficients = coefficients * self.dataset.data[self.eligible_col].to_numpy(dtype=bool)
        return coefficients

    def description(self):
        return f"""
        Retained basal area must be between {self.lb} and {self.ub} m2/ha.
        """


class MaxCuttingDiameter(ProportionConstraint):
    """Prevent thinning of trees whose DBH exceeds a maximum diameter."""

    indicator_col = 'CG_exceeds_max_cutting_diameter'

    def __init__(self, max_cutting_diameter: float = 0.0):
        self.max_cutting_diameter = max_cutting_diameter
        super().__init__(name="Maximum cutting diameter",
            lb=0.0, ub=0.0, prop_col=self.indicator_col, prop_vals={1},
            applies_to=DecisionScope.SELECT)

    def bind(self, dataset: SingleTreeDataset) -> "MaxCuttingDiameter":
        dbh_col = dataset._get_column_name('dbh')
        dataset.data[self.indicator_col] = (
            dataset.data[dbh_col] > self.max_cutting_diameter
        ).astype(np.int8)
        return super().bind(dataset)

    def description(self):
        return f"""
        Trees with diameter greater than {self.max_cutting_diameter} cannot be cut.
        """


class WaterProtectionConstraint(ProportionConstraint):
    """Prevent harvesting trees marked as inside a water-protection buffer."""

    indicator_col = 'CG_water_protected'

    def __init__(self):
        super().__init__(
            name="Water-body protection buffer",
            lb=0.0,
            ub=0.0,
            prop_col=self.indicator_col,
            prop_vals={True, 1},
            applies_to=DecisionScope.SELECT,
        )

    def bind(self, dataset: SingleTreeDataset) -> "WaterProtectionConstraint":
        if self.indicator_col not in dataset.data.columns:
            raise ValueError(
                f"Dataset is missing required column '{self.indicator_col}'."
            )
        return super().bind(dataset)

    def description(self):
        return "Trees inside the configured water buffer cannot be cut."


class RoadHarvestConstraint(WaterProtectionConstraint):
    """Require harvesting every tree marked inside the road buffer."""

    indicator_col = 'CG_road_harvest'

    def __init__(self):
        super().__init__()
        self.name = "Road clearing buffer"
        self.applies_to = DecisionScope.NOT_SELECT

    def description(self):
        return "Trees inside the configured road buffer must be harvested."

#########################################################
#              GROUP BALANCE CONSTRAINTS                #
#########################################################

class GroupBalanceConstraint(Constraint):
    """Enforce proportion bounds on groups of trees."""
    
    def __init__(self, name: str,
                 group_bounds: dict[tuple], group_col: str = None,
                 applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(name=name, applies_to=applies_to)
        self.type = 'group_balance'
        self.group_col = group_col
        self.group_bounds = group_bounds # {'oak': (0.1, 0.5), 'pine': (0.2, 0.6)}

    ################
    # CHECK GROUPS #
    ################

    @property
    def group_bounds(self):
        return self._group_bounds
    
    @group_bounds.setter
    def group_bounds(self, value):
        if not isinstance(value, dict):
            raise ValueError("`group_bounds` must be a dictionary of the form {group: (lb, ub)}.")
        for group, bounds in value.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Bounds for group '{group}' must be a tuple of (lb, ub).")
            lb, ub = bounds
            if not (0 <= lb <= ub <= 1):
                raise ValueError(f"Bounds for group '{group}' must satisfy 0 <= lb <= ub <= 1. \n Got {bounds}.")
        self._group_bounds = value

    def bind(self, dataset: SingleTreeDataset) -> "GroupBalanceConstraint":
        super().bind(dataset)
        if self.group_col is None:
            raise ValueError("A group-balance constraint requires a group column.")
        available = set(dataset.data[self.group_col].unique())
        missing = set(self.group_bounds).difference(available)
        if missing:
            raise ValueError(
                f"Groups {sorted(missing)!r} are not present in the dataset's "
                f"`{self.group_col}` column."
            )
        return self

    ###############
    # CALCULATION #
    ###############

    def evaluate(self, decisions: list) -> dict:
        """Evaluate proportion of decisions per group."""
        group_counts = self.dataset.data.groupby(self.group_col).size()
        decision_counts = self.dataset.data.iloc[self.decision_mask(decisions)].groupby(self.group_col).size()
        proportions = (decision_counts / group_counts).fillna(0)
        return proportions.to_dict()

    def is_satisfied(self, decisions: list) -> bool:
        """Constraint is satisfied if proportion of decisions per group is within bounds."""
        proportions = self.evaluate(decisions)
        return all(
            lb <= proportions.get(group, 0) <= ub
            for group, (lb, ub) in self.group_bounds.items()
        )

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        
        return f"""
        Proportion of decisions in every group of `{self.group_col}` must remain within its specified bounds.
        """
    
    @property
    def groups(self):
        """Get unique groups in the dataset."""
        return self.dataset.data[self.group_col].unique()

class SpeciesConstraint(GroupBalanceConstraint):
    """Ensure species balance in selected trees."""
    
    def __init__(self, group_bounds: dict[tuple] = None,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(name="Species balance constraint",
            group_bounds=group_bounds, group_col=None, applies_to=applies_to)

    def bind(self, dataset: SingleTreeDataset) -> "SpeciesConstraint":
        self.group_col = dataset._get_column_name('species')
        return super().bind(dataset)

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        
        return """
        Proportion of trees per species must be between specified bounds.
        """


class FrequencyConstraint(Constraint):
    """Keep the number of selected trees in each supplied bin within bounds."""

    def __init__(self, name: str, masks: list[np.ndarray], bounds: list[tuple[int, int]],
                 applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(name=name, applies_to=applies_to)
        self.type = "count"
        self.masks = masks
        self.bounds = bounds

    def bind(self, dataset: SingleTreeDataset) -> "FrequencyConstraint":
        super().bind(dataset)
        if len(self.masks) != len(self.bounds):
            raise ValueError("Frequency masks and bounds must have the same length.")
        if any(mask.shape != (dataset._n_trees,) for mask in self.masks):
            raise ValueError("Frequency masks must match the dataset length.")
        return self

    def evaluate(self, decisions: list) -> list[int]:
        selected = self.decision_mask(decisions)
        return [int((selected & mask).sum()) for mask in self.masks]

    def is_satisfied(self, decisions: list) -> bool:
        return all(
            lower <= count <= upper
            for count, (lower, upper) in zip(self.evaluate(decisions), self.bounds)
        )

    def description(self):
        return "Selected-tree frequencies must remain within the configured bounds."

class ZTreeSpatialConstraint(GroupBalanceConstraint):
    """Spatially distribute Z trees by enforcing even proportions among K clusters."""

    def __init__(self, lb_trees_per_ha: float=10,
        ub_trees_per_ha: float=20, n_clusters: int=5,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        self.lb_trees_per_ha = lb_trees_per_ha
        self.ub_trees_per_ha = ub_trees_per_ha
        self.n_clusters = n_clusters
        self.lb = None
        self.ub = None
        super().__init__(name="Z tree spatial constraint",
            group_bounds={}, group_col='CG_kmeans_cluster', applies_to=applies_to)

    def bind(self, dataset: SingleTreeDataset) -> "ZTreeSpatialConstraint":
        dataset = self.K_means_partition(dataset, self.n_clusters)
        self.lb = self.density_to_proportion(
            self.lb_trees_per_ha, dataset.hectares, dataset._n_trees
        )
        self.ub = self.density_to_proportion(
            self.ub_trees_per_ha, dataset.hectares, dataset._n_trees
        )
        self.group_bounds = {i: (self.lb, self.ub) for i in range(self.n_clusters)}
        return super().bind(dataset)

    ###############
    # CALCULATION #
    ###############

    @staticmethod
    def K_means_partition(dataset: SingleTreeDataset, n_clusters: int) -> SingleTreeDataset:
        """Partition trees into clusters using K-means."""

        if 'CG_kmeans_cluster' in dataset.data.columns:
            print("K-means clustering already performed. Using existing clusters.")
            return dataset

        # get tree coordinates
        x_col = dataset._get_column_name('x')
        y_col = dataset._get_column_name('y')
        coords = dataset.data[[x_col, y_col]].values

        # perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        dataset.data['CG_kmeans_cluster'] = kmeans.labels_
        return dataset

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        return f"""
        Proportion of Z trees in each of the {self.n_clusters} clusters must be between {self.lb} and {self.ub}.
        """

class SpatialGridConstraint(GroupBalanceConstraint):
    """Spatially distribute Z trees by enforcing one tree per grid square."""

    def __init__(self, grid_type: Literal['square', 'hexagon'] = 'square',
        grid_size: float = 10.0,
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        group_col = 'CG_grid_square_id'
        self.grid_type = grid_type
        self.grid_size = grid_size
        super().__init__(name="Spatial grid constraint",
            group_bounds={}, group_col=group_col, applies_to=applies_to)

    def bind(self, dataset: SingleTreeDataset) -> "SpatialGridConstraint":
        dataset = self.assign_grid_squares(dataset, self.grid_type, self.grid_size)
        self.group_bounds = self.calculate_group_bounds(dataset, self.group_col)
        return super().bind(dataset)

    ###############
    # CALCULATION #
    ###############

    @staticmethod
    def assign_grid_squares(dataset: SingleTreeDataset, grid_type: Literal['square', 'hexagon'], grid_size: float) -> SingleTreeDataset:
        """Assign trees to grid squares based on their coordinates."""
        if grid_type not in {'square', 'hexagon'}:
            raise ValueError("grid_type must be either 'square' or 'hexagon'.")
        if grid_size <= 0:
            raise ValueError("grid_size must be greater than 0.")

        dataset_metric = dataset.metric_conversion(dataset.data)
        x = dataset_metric.geometry.x.to_numpy(dtype=float)
        y = dataset_metric.geometry.y.to_numpy(dtype=float)
        if grid_type == 'square':
            grid_ids, grid_polygons = SpatialGridConstraint.calculate_grid_squares(x, y, grid_size)
        elif grid_type == 'hexagon':
            grid_ids, grid_polygons = SpatialGridConstraint.calculate_hexagons(x, y, grid_size)

        dataset.data.loc[dataset_metric.index, 'CG_grid_square_id'] = grid_ids
        dataset.grid_squares = gpd.GeoDataFrame(
            {
                'CG_grid_square_id': list(grid_polygons.keys()),
                'grid_type': grid_type,
                'grid_size': grid_size,
            },
            geometry=list(grid_polygons.values()),
            crs=dataset_metric.crs,
        )
        return dataset

    @staticmethod
    def calculate_grid_squares(x: np.ndarray, y: np.ndarray, grid_size: float) -> tuple[list[str], dict[str, Polygon]]:
        """Calculate square grid IDs and polygons for the supplied coordinates."""
        grid_x = np.floor(x / grid_size).astype(int)
        grid_y = np.floor(y / grid_size).astype(int)
        grid_ids = [f"square_{gx}_{gy}" for gx, gy in zip(grid_x, grid_y)]
        grid_polygons = {}

        for grid_id, gx, gy in zip(grid_ids, grid_x, grid_y):
            if grid_id not in grid_polygons:
                x_min = gx * grid_size
                y_min = gy * grid_size
                grid_polygons[grid_id] = box(x_min, y_min, x_min + grid_size, y_min + grid_size)

        return grid_ids, grid_polygons

    @staticmethod
    def calculate_hexagons(x: np.ndarray, y: np.ndarray, grid_size: float) -> tuple[list[str], dict[str, Polygon]]:
        """Calculate pointy-top hexagonal grid IDs and polygons."""
        # grid_size is the hex width in metres.
        side_length = grid_size / np.sqrt(3)
        q = ((np.sqrt(3) / 3 * x) - (1 / 3 * y)) / side_length
        r = (2 / 3 * y) / side_length
        rounded_hexes = [
            SpatialGridConstraint._round_axial_hex(q_i, r_i)
            for q_i, r_i in zip(q, r)
        ]
        grid_ids = [f"hex_{q_i}_{r_i}" for q_i, r_i in rounded_hexes]
        grid_polygons = {}

        for grid_id, (q_i, r_i) in zip(grid_ids, rounded_hexes):
            if grid_id not in grid_polygons:
                grid_polygons[grid_id] = SpatialGridConstraint._hex_polygon(q_i, r_i, side_length)

        return grid_ids, grid_polygons

    @staticmethod
    def _hex_polygon(q: int, r: int, side_length: float) -> Polygon:
        """Create a pointy-top hexagon polygon from axial grid coordinates."""
        center_x = side_length * np.sqrt(3) * (q + r / 2)
        center_y = side_length * 3 / 2 * r
        vertices = []
        for vertex in range(6):
            angle = np.pi / 180 * (30 + 60 * vertex)
            vertices.append((
                center_x + side_length * np.cos(angle),
                center_y + side_length * np.sin(angle),
            ))
        return Polygon(vertices)

    @staticmethod
    def _round_axial_hex(q: float, r: float) -> tuple[int, int]:
        """Round fractional axial hex coordinates to the nearest hex cell."""
        x = q
        z = r
        y = -x - z

        rx = round(x)
        ry = round(y)
        rz = round(z)

        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)

        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry

        return int(rx), int(rz)

    @staticmethod
    def calculate_group_bounds(dataset: SingleTreeDataset, group_col: str) -> dict:
        """Calculate group bounds for grid squares."""
        value_counts = dataset.data[group_col].value_counts()
        epsilon = 0.5 / dataset._n_trees
        group_bounds = {group: (0, min((1/count) + epsilon, 1)) for group, count in value_counts.items()}
        return group_bounds

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        return f"""
        Each grid square of size {self.grid_size}x{self.grid_size} must contain at most one selected tree.
        """



#########################################################
#           PAIRWISE DISTANCE CONSTRAINTS               #
#########################################################

class PairwiseDistanceConstraint(Constraint):
    """Ensure minimum distance between selected trees."""
    
    def __init__(self, min_distance: float = 5.0,
        applies_to: DecisionScope | str = DecisionScope.SELECT,
        exclude_col: str | None = None,
        exclude_vals: set | None = None):
        super().__init__(name="Pairwise distance constraint",
            applies_to=applies_to)
        self.min_distance = min_distance
        self.exclude_col = exclude_col
        self.exclude_vals = {True, 1} if exclude_vals is None else set(exclude_vals)
        self.max_distance = None
        self.type = 'pairwise_distance'

    def eligible_mask(self, dataset: SingleTreeDataset) -> np.ndarray:
        """Return trees that participate in pairwise spacing checks."""
        if self.exclude_col is None or self.exclude_col not in dataset.data.columns:
            return np.ones(dataset._n_trees, dtype=bool)
        return ~dataset.data[self.exclude_col].isin(self.exclude_vals).to_numpy(dtype=bool)

    ###############
    # CALCULATION #
    ###############

    def calculate_distance_matrix(self, dataset: SingleTreeDataset) -> np.ndarray:
        """Calculate pairwise distance matrix between trees."""

        dataset_metric = dataset.metric_conversion(dataset.data)
        coordinates = np.column_stack((dataset_metric.geometry.x.to_numpy(dtype=float),
                                       dataset_metric.geometry.y.to_numpy(dtype=float)))
        distance_matrix = pairwise_distances(coordinates, metric='euclidean')

        max_distance = np.max(distance_matrix)
        self.max_distance = max_distance

        # ignore self-distances
        np.fill_diagonal(distance_matrix, max_distance + 1)
        
        return distance_matrix

    def conflicting_pairs(self, dataset: SingleTreeDataset) -> np.ndarray:
        """Return tree pairs that violate the minimum distance constraint."""
        dataset_metric = dataset.metric_conversion(dataset.data)
        eligible_positions = np.flatnonzero(self.eligible_mask(dataset))
        if eligible_positions.size < 2:
            return np.empty((0, 2), dtype=int)
        coordinates = np.column_stack((
            dataset_metric.geometry.x.to_numpy(dtype=float)[eligible_positions],
            dataset_metric.geometry.y.to_numpy(dtype=float)[eligible_positions],
        ))
        pairs = cKDTree(coordinates).query_pairs(
            self.min_distance, output_type="ndarray"
        )
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)

        # query_pairs includes points exactly on the radius; the constraint is
        # violated only when distance is strictly less than min_distance.
        deltas = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
        pairs = pairs[np.einsum("ij,ij->i", deltas, deltas)
                      < self.min_distance ** 2]
        return eligible_positions[pairs]

    def evaluate(self, decisions: list) -> float:
        """Evaluate minimum pairwise distance among selected trees."""
        distance_matrix = self.calculate_distance_matrix(self.dataset)
        indices = np.flatnonzero(
            self.decision_mask(decisions) & self.eligible_mask(self.dataset)
        )
        if indices.size < 2:
            return float("inf")
        min_dist = np.min(distance_matrix[np.ix_(indices, indices)])
        return min_dist

    def is_satisfied(self, decisions: list) -> bool:
        """Constraint is satisfied if minimum pairwise distance is above threshold."""
        min_dist = self.evaluate(decisions)
        return min_dist >= self.min_distance

    ###############
    # DESCRIPTION #
    ###############

    def description(self):
        return f"""
        Minimum pairwise distance between selected trees must be at least {self.min_distance} units.
        """
