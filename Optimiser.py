from SingleTreeDataset import *
from Constraints import *
from Objectives import *
from SingleTreeAlgorithms import *
from DecisionScope import DecisionScope
from time import perf_counter
import numpy as np

class TreeOptimiser:
    """Shared optimiser for a positional binary tree-decision vector."""

    treatment_type = "generic"
    select_label = "selected"
    not_select_label = "not selected"
    
    def __init__(self, dataset: SingleTreeDataset, objectives: dict[Objective, float], constraints: list[Constraint], algorithm: SingleTreeAlgorithm):
        self.dataset = dataset
        self.objectives = list(objectives.keys())
        self.constraints = constraints
        self.algorithm = algorithm
        self.obj_weights = list(objectives.values())
        self.senses = [objective.sense for objective in self.objectives]
        self.objective_ranges = {}

        self._bind_problem_definition()
        self._validate_problem_definition()

    def _bind_problem_definition(self) -> None:
        """Bind reusable objective and constraint definitions to this dataset."""
        for component in [*self.objectives, *self.constraints]:
            component.bind(self.dataset)

    def _validate_problem_definition(self) -> None:
        """Reject ambiguous or cross-dataset model components before solving."""
        for component in [*self.objectives, *self.constraints]:
            if component.dataset is not self.dataset:
                raise ValueError(
                    "Every objective and constraint must refer to the optimiser's "
                    "dataset instance."
                )
            if component.applies_to not in {
                DecisionScope.SELECT, DecisionScope.NOT_SELECT
            }:
                raise ValueError("Each component must apply to SELECT or NOT_SELECT.")

    ########################
    # PERFORM OPTIMISATION #
    ########################

    def assign_hyperparameters(self, algorithm_hyperparameters: dict = None):
        """Assign hyperparameters to the algorithm."""
        if algorithm_hyperparameters is None:
            return

        for key, value in algorithm_hyperparameters.items():
            if hasattr(self.algorithm, key):
                setattr(self.algorithm, key, value)
            else:
                raise ValueError(f"Algorithm {self.algorithm.name} does not have a hyperparameter named '{key}'.")

    def optimise(self, algorithm_hyperparameters: dict = None) -> dict:
        """Optimise tree selection."""
        optimise_started = perf_counter()
        self.assign_hyperparameters(algorithm_hyperparameters)
        normalization_start = perf_counter()
        if len(self.obj_weights) == 1:
            # Scaling a single objective by its feasible range cannot change the
            # optimum, so avoid two unnecessary LP solves for its min/max.
            normalized_weights = self.obj_weights.copy()
        else:
            self.calculate_objective_ranges()
            normalized_weights = self.normalized_objective_weights()
        normalization_seconds = perf_counter() - normalization_start

        start_time = perf_counter()
        results = self.algorithm.solve(
            self.dataset, self.objectives, self.constraints,
            normalized_weights, self.senses
        )
        results["solve_seconds"] = perf_counter() - start_time
        results["normalization_seconds"] = normalization_seconds
        results["objective_ranges"] = self.objective_ranges
        results["total_seconds"] = perf_counter() - optimise_started
        return self._annotate_result(results)

    def _annotate_result(self, results: dict) -> dict:
        """Attach explicit binary-decision semantics to an algorithm result."""
        vector = self.dataset.validate_decision_vector(results["decision_vector"])
        results["decision_vector"] = vector
        results["select_indices"] = self.dataset.decision_indices(
            vector, DecisionScope.SELECT
        )
        results["not_select_indices"] = self.dataset.decision_indices(
            vector, DecisionScope.NOT_SELECT
        )
        results["action_indices"] = results["select_indices"]
        results["treatment_type"] = self.treatment_type
        results["select_label"] = self.select_label
        results["not_select_label"] = self.not_select_label
        return results

    def calculate_objective_ranges(self,
        algorithm_hyperparameters: dict = None) -> dict:
        """Calculate and cache exact feasible LP ranges for all objectives."""
        if self.objective_ranges:
            return self.objective_ranges

        range_algorithm = LinearProgrammingAlgorithm()
        if isinstance(self.algorithm, LinearProgrammingAlgorithm):
            for key in ("solver_name", "max_time", "num_workers"):
                setattr(range_algorithm, key, getattr(self.algorithm, key))
        if algorithm_hyperparameters is not None:
            for key, value in algorithm_hyperparameters.items():
                if hasattr(range_algorithm, key):
                    setattr(range_algorithm, key, value)
                else:
                    raise ValueError(
                        f"Algorithm {range_algorithm.name} does not have a "
                        f"hyperparameter named '{key}'."
                    )

        for objective in self.objectives:
            minimum_result = range_algorithm.solve_single_objective(
                self.dataset, objective, self.constraints, sense="min"
            )
            maximum_result = range_algorithm.solve_single_objective(
                self.dataset, objective, self.constraints, sense="max"
            )

            for direction, result in (
                ("minimum", minimum_result), ("maximum", maximum_result)
            ):
                if not result.get("is_optimal", False):
                    raise RuntimeError(
                        f"Could not calculate the exact {direction} for objective "
                        f"'{objective.name}': {result['status']}"
                    )

            minimum = self.score_objective(objective, minimum_result["decision_vector"])
            maximum = self.score_objective(objective, maximum_result["decision_vector"])
            self.objective_ranges[objective.name] = {
                "minimum": minimum,
                "maximum": maximum,
                "minimum_status": minimum_result["status"],
                "maximum_status": maximum_result["status"],
                "minimum_decisions": minimum_result["decisions"],
                "maximum_decisions": maximum_result["decisions"],
            }

        return self.objective_ranges

    def normalized_objective_weights(self) -> list[float]:
        """Scale weights by each objective's exact feasible value range."""
        self.calculate_objective_ranges()
        normalized_weights = []
        for objective, weight in zip(
            self.objectives, self.obj_weights, strict=True
        ):
            objective_range = self.objective_ranges[objective.name]
            value_range = (
                objective_range["maximum"] - objective_range["minimum"]
            )
            normalized_weights.append(
                0.0 if value_range == 0 else weight / value_range
            )
        return normalized_weights

    def score_objective(self, objective: Objective, decisions) -> float:
        """Calculate one scope-aware objective score from a binary vector.

        A positional list of selected indices is accepted temporarily for
        compatibility with existing callers, but algorithms always use vectors.
        """
        values = np.asarray(decisions)
        if values.ndim == 1 and values.size == self.dataset._n_trees and np.all(
            np.isin(values, (0, 1))
        ):
            decision_vector = self.dataset.validate_decision_vector(values)
        else:
            decision_vector = np.zeros(self.dataset._n_trees, dtype=np.int8)
            decision_vector[values.astype(int, copy=False)] = 1
        return float(objective.calculate(decision_vector))

    def benchmark_objectives(self, decisions: list[int],
        algorithm_hyperparameters: dict = None) -> dict:
        """Score a solution against the cached exact feasible LP ranges."""
        self.calculate_objective_ranges(algorithm_hyperparameters)
        benchmarks = {}

        for objective in self.objectives:
            actual = self.score_objective(objective, decisions)
            objective_range = self.objective_ranges[objective.name]
            if objective.sense == "max":
                best = objective_range["maximum"]
                worst = objective_range["minimum"]
                best_prefix = "maximum"
                worst_prefix = "minimum"
            else:
                best = objective_range["minimum"]
                worst = objective_range["maximum"]
                best_prefix = "minimum"
                worst_prefix = "maximum"
            percent = None if best == worst else 100 * (actual - worst) / (best - worst)

            benchmarks[objective.name] = {
                "actual": actual,
                "best": best,
                "worst": worst,
                "percent": percent,
                "best_status": objective_range[f"{best_prefix}_status"],
                "worst_status": objective_range[f"{worst_prefix}_status"],
                "best_decisions": objective_range[f"{best_prefix}_decisions"],
                "worst_decisions": objective_range[f"{worst_prefix}_decisions"],
            }

        return benchmarks

class FutureCropTreeOptimiser(TreeOptimiser):
    """Optimise future-crop selection where ``SELECT`` marks crop trees."""

    treatment_type = "future_crop_tree_selection"
    select_label = "selected as future crop tree"
    not_select_label = "not selected as future crop tree"

    def _annotate_result(self, results: dict) -> dict:
        results = super()._annotate_result(results)
        results["future_crop_tree_indices"] = results["select_indices"]
        self.future_crop_tree_indices = results["future_crop_tree_indices"]
        return results

    def calculate_competition_indices(self):
        """Calculate competition indices for the current future-crop trees."""
        if not hasattr(self, "future_crop_tree_indices"):
            raise RuntimeError(
                "Future crop trees must be selected before calculating "
                "competition indices."
            )
        return self.dataset.calculate_hegyi_indices(
            self.future_crop_tree_indices
        )

    def thinning_schedule(self, targets: list, target_metric: str = "density"):
        """Assign non-future-crop trees to successive thinning treatments.

        Density targets are numbers of stems remaining. Basal-area targets are
        m2/ha remaining. Trees with the highest competition indices are assigned
        first.
        """
        if target_metric not in {"density", "basal_area"}:
            raise ValueError("target_metric must be 'density' or 'basal_area'.")
        competition_indices, nearest_z_tree_indices = (
            self.dataset.calculate_hegyi_indices_with_nearest_z_tree(
                self.future_crop_tree_indices
            )
        )
        self.competition_indices = competition_indices
        self.nearest_z_tree_indices = nearest_z_tree_indices
        future_crop_tree_indices = set(self.future_crop_tree_indices)
        ranked_indices = [
            int(index)
            for index in np.argsort(competition_indices)[::-1]
            if index not in future_crop_tree_indices
        ]

        schedule = []
        ranked_offset = 0
        n_stems_standing = self.dataset._n_trees
        basal_areas_per_hectare = None
        basal_area_standing = None
        if target_metric == "basal_area":
            dbh_col = self.dataset._get_column_name("dbh")
            diameter_metres = self.dataset.data[dbh_col].to_numpy(dtype=float) / 100
            basal_areas_per_hectare = (
                np.pi * (diameter_metres / 2) ** 2 / self.dataset.hectares
            )
            basal_area_standing = float(basal_areas_per_hectare.sum())

        for thinning_round, target in enumerate(targets, start=1):
            if target_metric == "basal_area":
                n_trees_to_remove = 0
                while (
                    basal_area_standing > float(target)
                    and ranked_offset + n_trees_to_remove < len(ranked_indices)
                ):
                    index = ranked_indices[ranked_offset + n_trees_to_remove]
                    basal_area_standing -= basal_areas_per_hectare[index]
                    n_trees_to_remove += 1
            else:
                n_trees_to_remove = max(0, n_stems_standing - int(target))
            treatment_indices = ranked_indices[
                ranked_offset:ranked_offset + n_trees_to_remove
            ]
            schedule.append({
                "thinning_round": thinning_round,
                "n_stem_target": n_stems_standing - len(treatment_indices),
                "target": float(target),
                "target_metric": target_metric,
                "tree_indices": treatment_indices,
            })
            ranked_offset += len(treatment_indices)
            n_stems_standing -= len(treatment_indices)

        return schedule


class ThinningOptimiser(TreeOptimiser):
    """Optimise thinning where ``SELECT`` marks trees selected for removal."""

    treatment_type = "thinning_treatment"
    select_label = "selected for thinning (cut)"
    not_select_label = "not selected for thinning (left standing)"

    def _annotate_result(self, results: dict) -> dict:
        results = super()._annotate_result(results)
        results["cut_indices"] = results["select_indices"]
        results["retained_indices"] = results["not_select_indices"]
        return results


# Existing users of the tool receive the unchanged future-crop semantics.
ZTreeOptimiser = FutureCropTreeOptimiser

        
