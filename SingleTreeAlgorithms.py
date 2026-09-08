from SingleTreeDataset import SingleTreeDataset
from Constraints import Constraint
from Objectives import Objective
from DecisionScope import DecisionScope
from GurobiLicense import current_gurobi_environment, GurobiLicenseError
from abc import ABC, abstractmethod
from functools import partial
from time import perf_counter
from typing import Literal
from ortools.linear_solver import pywraplp

import numpy as np
import scipy.optimize
import scipy.sparse
import pygad


def _weighted_objective(x: np.ndarray, weighted_coefficients: np.ndarray) -> float | np.ndarray:
    """Evaluate one candidate, or a vectorized batch with shape (N, S)."""
    values = -np.asarray(weighted_coefficients) @ np.asarray(x)
    return float(values) if np.ndim(values) == 0 else values


def _aggregate_constraint_violation(x: np.ndarray,
    coefficients: scipy.sparse.spmatrix, lower_bounds: np.ndarray,
    upper_bounds: np.ndarray) -> np.ndarray:
    """Return one unnormalized violation total for a candidate or population."""
    values = coefficients @ np.asarray(x)

    if values.ndim == 1:
        lower_violation = np.maximum(lower_bounds - values, 0)
        upper_violation = np.maximum(values - upper_bounds, 0)
        return np.asarray([np.sum(lower_violation + upper_violation)])

    lower_violation = np.maximum(lower_bounds[:, np.newaxis] - values, 0)
    upper_violation = np.maximum(values - upper_bounds[:, np.newaxis], 0)
    return np.sum(lower_violation + upper_violation, axis=0, keepdims=True)

def _compute_weighted_coefficients(dataset: SingleTreeDataset,
    objectives: list[Objective], obj_weights: list[float],
    sense: list[Literal["max", "min"]]) -> np.ndarray:
    """Compute weighted coefficients for the objective function (weight * column value)."""
    
    weighted_coefficients = np.zeros(dataset._n_trees, dtype=float)
    
    for objective, weight, objective_sense in zip(
        objectives, obj_weights, sense, strict=True
    ):
        
        if objective.type != 'sum_product':
            raise NotImplementedError("Differential evolution only supports sum-product objectives.")
        
        coefficients = dataset.data[objective.col_name].to_numpy(
            dtype=float, copy=False
        )
        if objective.applies_to is DecisionScope.NOT_SELECT:
            coefficients = -coefficients
        weighted_coefficients += (
            (weight if objective_sense == "max" else -weight)
            * coefficients
            )
        
    return weighted_coefficients


def _build_constraint_system(
    dataset: SingleTreeDataset,
    constraints: list[Constraint],
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build one sparse linear constraint system without scalar DataFrame access."""
    rows = []
    lower_bounds = []
    upper_bounds = []

    def append_scoped_rows(
        target_rows: scipy.sparse.csr_matrix,
        target_lower: np.ndarray,
        target_upper: np.ndarray,
        constraint: Constraint,
    ) -> None:
        """Convert target-set rows to the canonical SELECT decision vector."""
        if constraint.applies_to is DecisionScope.NOT_SELECT:
            constants = np.asarray(target_rows.sum(axis=1)).ravel()
            target_rows = -target_rows
            target_lower = target_lower - constants
            target_upper = target_upper - constants
        rows.append(target_rows)
        lower_bounds.extend(target_lower)
        upper_bounds.extend(target_upper)

    for constraint in constraints:
        if constraint.type == 'proportion':
            append_scoped_rows(
                scipy.sparse.csr_matrix(
                    constraint.linear_coefficients()[np.newaxis, :]
                ),
                np.asarray([constraint.lb]), np.asarray([constraint.ub]), constraint,
            )

        elif constraint.type == 'group_balance':
            for group, (lb, ub) in constraint.group_bounds.items():
                group_mask = dataset.data[constraint.group_col].eq(group).to_numpy(
                    dtype=float, copy=False
                )
                group_size = int(group_mask.sum())
                if group_size == 0:
                    raise ValueError(
                        f"Group '{group}' for column '{constraint.group_col}' "
                        "has no trees in the dataset."
                    )
                append_scoped_rows(
                    scipy.sparse.csr_matrix(group_mask[np.newaxis, :] / group_size),
                    np.asarray([lb]), np.asarray([ub]), constraint,
                )

        elif constraint.type == 'count':
            for mask, (lb, ub) in zip(constraint.masks, constraint.bounds):
                append_scoped_rows(
                    scipy.sparse.csr_matrix(mask.astype(float)[np.newaxis, :]),
                    np.asarray([lb]), np.asarray([ub]), constraint,
                )

        elif constraint.type == 'pairwise_distance':
            if hasattr(constraint, "conflicting_pairs"):
                pairs = constraint.conflicting_pairs(dataset)
                conflicting_i = pairs[:, 0]
                conflicting_j = pairs[:, 1]
            else:
                distance_matrix = constraint.calculate_distance_matrix(dataset)
                conflicting_i, conflicting_j = np.where(
                    np.triu(distance_matrix < constraint.min_distance, k=1)
                )
            if conflicting_i.size:
                n_conflicts = conflicting_i.size
                row_indices = np.repeat(np.arange(n_conflicts), 2)
                column_indices = np.column_stack(
                    (conflicting_i, conflicting_j)
                ).ravel()
                target_rows = scipy.sparse.csr_matrix(
                    (np.ones(row_indices.size), (row_indices, column_indices)),
                    shape=(n_conflicts, dataset._n_trees),
                )
                append_scoped_rows(
                    target_rows, np.zeros(n_conflicts), np.ones(n_conflicts),
                    constraint,
                )

    return (
        scipy.sparse.vstack(rows, format="csr") if rows else
            scipy.sparse.csr_matrix((0, dataset._n_trees), dtype=float),
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )


def _constraint_violation(
    x: np.ndarray,
    coefficients: scipy.sparse.spmatrix,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    """Return the aggregate violation of a sparse linear constraint system."""
    if coefficients.shape[0] == 0:
        return 0.0
    values = coefficients @ x
    return float(np.sum(
        np.maximum(lower_bounds - values, 0)
        + np.maximum(values - upper_bounds, 0)
    ))


def _constraints_satisfied(
    x: np.ndarray,
    coefficients: scipy.sparse.spmatrix,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    """Check sparse linear constraints without calculating a violation total."""
    if coefficients.shape[0] == 0:
        return True
    values = coefficients @ x
    return not (
        np.any(values < lower_bounds - tolerance)
        or np.any(values > upper_bounds + tolerance)
    )


def _constraint_penalty_scale(weighted_coefficients: np.ndarray) -> float:
    """Return a scale that makes feasible solutions preferable to infeasible ones."""
    objective_bound = float(np.sum(np.abs(weighted_coefficients)))
    return 2 * objective_bound + 1


def _ortools_status_message(status: int) -> str:
    """Translate an OR-Tools MPSolver status code into a readable message."""
    messages = {
        pywraplp.Solver.OPTIMAL: "Optimal solution found.",
        pywraplp.Solver.FEASIBLE: "Feasible solution found; optimality was not proven.",
        pywraplp.Solver.INFEASIBLE: "No feasible solution exists.",
        pywraplp.Solver.UNBOUNDED: "The model is unbounded.",
        pywraplp.Solver.ABNORMAL: "The solver stopped because an abnormal error occurred.",
        pywraplp.Solver.NOT_SOLVED: "The solver stopped before finding a solution.",
    }
    model_invalid = getattr(pywraplp.Solver, "MODEL_INVALID", None)
    if model_invalid is not None:
        messages[model_invalid] = "The optimization model is invalid."
    return messages.get(status, f"Solver finished with status code {status}.")


def _gurobi_status_message(status: int, solution_count: int) -> str:
    """Translate a Gurobi status code into a readable message."""
    try:
        import gurobipy as gp
    except ImportError:
        return f"Gurobi finished with status code {status}."

    messages = {
        gp.GRB.LOADED: "Model loaded but not optimized.",
        gp.GRB.OPTIMAL: "Optimal solution found.",
        gp.GRB.INFEASIBLE: "No feasible solution exists.",
        gp.GRB.INF_OR_UNBD: "The model is infeasible or unbounded.",
        gp.GRB.UNBOUNDED: "The model is unbounded.",
        gp.GRB.CUTOFF: "The objective cutoff was reached without a qualifying solution.",
        gp.GRB.ITERATION_LIMIT: "Iteration limit reached.",
        gp.GRB.NODE_LIMIT: "Node limit reached.",
        gp.GRB.TIME_LIMIT: "Time limit reached.",
        gp.GRB.SOLUTION_LIMIT: "Solution limit reached.",
        gp.GRB.INTERRUPTED: "Optimization was interrupted.",
        gp.GRB.NUMERIC: "Optimization stopped because of numerical difficulties.",
        gp.GRB.SUBOPTIMAL: "A suboptimal solution was returned.",
        gp.GRB.INPROGRESS: "Optimization is still in progress.",
        gp.GRB.USER_OBJ_LIMIT: "User objective limit reached.",
        gp.GRB.WORK_LIMIT: "Work limit reached.",
        gp.GRB.MEM_LIMIT: "Memory limit reached.",
    }
    message = messages.get(status, f"Gurobi finished with status code {status}.")
    if solution_count and status != gp.GRB.OPTIMAL:
        message = f"{message} A feasible solution is available."
    return message

class SingleTreeAlgorithm(ABC):
    """Base class for single tree optimisation algorithms."""
    
    def __init__(self, name: str):
        
        self.name = name
        self._constraint_cache_key = None
        self._constraint_cache = None

    def _constraint_system(self, dataset: SingleTreeDataset,
        constraints: list[Constraint]) -> tuple[
            scipy.sparse.csr_matrix, np.ndarray, np.ndarray
        ]:
        """Reuse compiled constraints across repeated solves on the same model."""
        cache_key = (id(dataset), tuple(id(constraint) for constraint in constraints))
        if cache_key != self._constraint_cache_key:
            self._constraint_cache = _build_constraint_system(dataset, constraints)
            self._constraint_cache_key = cache_key
        return self._constraint_cache

    #########
    # SOLVE #
    #########

    def solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:
        """Validate objective directions and solve the weighted maximization."""
        solve_started = perf_counter()
        if len(sense) != len(objectives):
            raise ValueError("sense must contain one value per objective.")
        if any(value not in {"max", "min"} for value in sense):
            raise ValueError("Each sense must be either 'max' or 'min'.")
        result = self._solve(
            dataset=dataset, objectives=objectives, constraints=constraints,
            obj_weights=obj_weights, sense=sense
        )
        result.setdefault("timing", {})["solve_total_seconds"] = (
            perf_counter() - solve_started
        )
        if "decision_vector" in result:
            vector = dataset.validate_decision_vector(result["decision_vector"])
        else:
            vector = np.zeros(dataset._n_trees, dtype=np.int8)
            vector[np.asarray(result.get("decisions", []), dtype=int)] = 1
        result["decision_vector"] = vector
        result["decisions"] = np.flatnonzero(vector).tolist()
        return result

    @abstractmethod
    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:
        """Calculate optimal tree selection for a given dataset."""
        pass

    def solve_single_objective(self, dataset: SingleTreeDataset, objective: Objective,
        constraints: list[Constraint],
        sense: Literal["max", "min"] = "max") -> dict:
        """Solve a constrained optimisation problem for one objective."""
        return self.solve(dataset=dataset, objectives=[objective], constraints=constraints,
            obj_weights=[1.0], sense=[sense])

    ###############
    # DESCRIPTION #
    ###############

    @abstractmethod
    def _description(self):
        """Description of the algorithm."""
        pass

    def describe(self):
        """Print description of the algorithm."""
        print(f"Algorithm: {self.name}\n")
        print(self._description())

class LinearProgrammingAlgorithm(SingleTreeAlgorithm):
    """Linear programming algorithm for single tree optimisation."""
    
    def __init__(self):
        super().__init__(name="Linear Programming")

        self.solver_name = "CBC"  # Included solver; Gurobi remains an explicit option.
        self.max_time = 120
        self.num_workers = 1

    #########
    # SOLVE #
    #########

    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float], 
        sense: list[Literal["max", "min"]]) -> dict:
        """Compute optimal binary 0-1 future-crop tree selecton."""

        setup_started = perf_counter()

        if self.solver_name == "GUROBI":
            return self._solve_with_gurobi(
                dataset, objectives, constraints, obj_weights, sense
            )

        # instantiate model
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        if solver is None:
            raise RuntimeError(
                f"The '{self.solver_name}' solver is unavailable. For Gurobi, "
                "ensure Gurobi is installed and its license is active."
            )

        # decision variables: 1 = select tree, 0 = do not select tree
        x = [solver.BoolVar(f'{i}') for i in range(dataset._n_trees)]

        coefficients, lower_bounds, upper_bounds = self._constraint_system(
            dataset, constraints
        )
        for row_index in range(coefficients.shape[0]):
            start = coefficients.indptr[row_index]
            end = coefficients.indptr[row_index + 1]
            expression = solver.Sum(
                float(value) * x[int(column)]
                for column, value in zip(
                    coefficients.indices[start:end], coefficients.data[start:end]
                )
            )
            solver.Add(expression >= float(lower_bounds[row_index]))
            solver.Add(expression <= float(upper_bounds[row_index]))

        weighted_coefficients = _compute_weighted_coefficients(
            dataset, objectives, obj_weights, sense
        )

        # set objective
        objective_expr = solver.Sum(
            float(coefficient) * variable
            for coefficient, variable in zip(weighted_coefficients, x)
            if coefficient != 0
        )
        solver.Maximize(objective_expr)

        # solve
        solver.SetTimeLimit(int(self.max_time * 1000))
        solver.SetNumThreads(self.num_workers)
        engine_started = perf_counter()
        status = solver.Solve()
        engine_finished = perf_counter()

        # get decisions
        decisions = [i for i in range(dataset._n_trees) if x[i].solution_value() > 0.5]

        # return results
        result = {
            "status": _ortools_status_message(status),
            "status_code": status,
            "is_optimal": status == pywraplp.Solver.OPTIMAL,
            "decisions": decisions,
        }
        result["timing"] = {
            "engine_call": "ortools.Solver.Solve",
            "setup_seconds": engine_started - setup_started,
            "engine_seconds": engine_finished - engine_started,
            "postprocess_seconds": perf_counter() - engine_finished,
        }
        return result

    def _solve_with_gurobi(self, dataset: SingleTreeDataset,
        objectives: list[Objective], constraints: list[Constraint],
        obj_weights: list[float], sense: list[Literal["max", "min"]]) -> dict:
        """Solve the binary linear model through the native Gurobi API."""
        setup_started = perf_counter()
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError(
                "Gurobi was selected, but the 'gurobipy' package is not installed."
            ) from exc

        try:
            with gp.Model("SingleTreeOpt", env=current_gurobi_environment()) as model:
                model.Params.TimeLimit = float(self.max_time)
                model.Params.Threads = int(self.num_workers)
                x = model.addMVar(dataset._n_trees, vtype=gp.GRB.BINARY, name="tree")

                coefficients, lower_bounds, upper_bounds = (
                    self._constraint_system(dataset, constraints)
                )
                if coefficients.shape[0]:
                    values = coefficients @ x
                    model.addConstr(values >= lower_bounds, name="constraint_lb")
                    model.addConstr(values <= upper_bounds, name="constraint_ub")

                weighted_coefficients = _compute_weighted_coefficients(
                    dataset, objectives, obj_weights, sense
                )
                model.setObjective(weighted_coefficients @ x, gp.GRB.MAXIMIZE)
                engine_started = perf_counter()
                model.optimize()
                engine_finished = perf_counter()

                decisions = []
                if model.SolCount:
                    decisions = np.flatnonzero(np.asarray(x.X) > 0.5).tolist()
                result = {
                    "status": _gurobi_status_message(model.Status, model.SolCount),
                    "status_code": model.Status,
                    "is_optimal": model.Status == gp.GRB.OPTIMAL,
                    "decisions": decisions,
                }
                result["timing"] = {
                    "engine_call": "gurobipy.Model.optimize",
                    "setup_seconds": engine_started - setup_started,
                    "engine_seconds": engine_finished - engine_started,
                    "postprocess_seconds": perf_counter() - engine_finished,
                }
                return result
        except gp.GurobiError:
            raise GurobiLicenseError(
                "Gurobi could not solve the model. Check the WLS licence's validity, "
                "limits and support for this cloud host, or select CBC."
            ) from None
    
    ###############
    # DESCRIPTION #
    ###############

    def _description(self):
        return "Formulate as a linear program and solve with CBC."

class DifferentialEvolutionAlgorithm(SingleTreeAlgorithm):
    """Differential Evolution algorithm for single tree optimisation."""

    def __init__(self):
        super().__init__(name="Differential Evolution")

        self.maxiter = 1000
        self.popsize = 15
        self.mutation = (0.5, 1)
        self.tol = 1e-2
        self.random_seed = None
        self.progress_callback = None

    #########
    # SOLVE #
    #########

    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:

        setup_started = perf_counter()
        weighted_coefficients = _compute_weighted_coefficients(
            dataset, objectives, obj_weights, sense
        )

        f = partial(
            _weighted_objective,
            weighted_coefficients=weighted_coefficients,
        )
        
        bounds = np.tile((0.0, 1.0), (dataset._n_trees, 1))
        integrality = np.ones(dataset._n_trees, dtype=bool)
        combined_coefficients, combined_lower_bounds, combined_upper_bounds = (
            self._constraint_system(dataset, constraints)
        )
        if combined_coefficients.shape[0]:
            aggregate_violation = partial(
                _aggregate_constraint_violation,
                coefficients=combined_coefficients,
                lower_bounds=combined_lower_bounds,
                upper_bounds=combined_upper_bounds,
            )
            scipy_constraints = (
                scipy.optimize.LinearConstraint(
                    combined_coefficients,
                    combined_lower_bounds,
                    combined_upper_bounds,
                ),
            )
        else:
            aggregate_violation = None
            scipy_constraints = ()

        engine_started = perf_counter()
        result = scipy.optimize.differential_evolution(
            f,
            bounds=bounds,
            constraints=scipy_constraints,
            integrality=integrality,
            workers=1,
            vectorized=True,
            maxiter=self.maxiter,
            popsize=self.popsize,
            mutation=self.mutation,
            tol=self.tol,
            rng=self.random_seed,
            callback=self.progress_callback,
        )
        engine_finished = perf_counter()

        decisions = np.flatnonzero(result.x > 0.5).tolist()
        constraint_violation = (
            0.0 if aggregate_violation is None
            else float(aggregate_violation(result.x)[0])
        )
        
        result_data = {
            "status": result.message,
            "decisions": decisions,
            "constraint_violation": constraint_violation,
            "is_feasible": constraint_violation == 0,
        }
        result_data["timing"] = {
            "engine_call": "scipy.optimize.differential_evolution",
            "setup_seconds": engine_started - setup_started,
            "engine_seconds": engine_finished - engine_started,
            "postprocess_seconds": perf_counter() - engine_finished,
        }
        return result_data

    def _description(self):
        return "Formulate as differential evolution and solve using SciPy."

class BruteForceAlgorithm(SingleTreeAlgorithm):
    """Brute force algorithm for single tree optimisation."""

    def __init__(self):
        super().__init__(name="Brute Force")

        self.Ns = 2

    #########
    # SOLVE #
    #########

    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:
        """Exhaustively evaluate every feasible binary tree selection."""

        setup_started = perf_counter()
        weighted_coefficients = _compute_weighted_coefficients(
            dataset, objectives, obj_weights, sense
        )

        f = partial(
            _weighted_objective,
            weighted_coefficients=weighted_coefficients,
        )

        coefficients, lower_bounds, upper_bounds = (
            self._constraint_system(dataset, constraints)
        )

        def constrained_objective(x: np.ndarray) -> float:
            """Penalise infeasible points on SciPy's binary search grid."""
            x = np.asarray(x)
            if not _constraints_satisfied(
                x, coefficients, lower_bounds, upper_bounds
            ):
                return np.inf

            return f(x)

        engine_started = perf_counter()
        best_x, best_value, _, _ = scipy.optimize.brute(
            constrained_objective,
            ranges=((0, 1),) * dataset._n_trees,
            Ns=self.Ns,
            full_output=True,
            finish=None,
        )
        engine_finished = perf_counter()
        best_x = np.rint(np.atleast_1d(best_x)).astype(np.int8)

        if not np.isfinite(best_value):
            result_data = {
                "status": "No feasible solution found.",
                "decisions": [],
                "constraint_violation": None,
                "is_feasible": False,
            }
            result_data["timing"] = {
                "engine_call": "scipy.optimize.brute",
                "setup_seconds": engine_started - setup_started,
                "engine_seconds": engine_finished - engine_started,
                "postprocess_seconds": perf_counter() - engine_finished,
            }
            return result_data

        result_data = {
            "status": "Optimal solution found by exhaustive search.",
            "decisions": np.flatnonzero(best_x).tolist(),
            "constraint_violation": 0.0,
            "is_feasible": True,
        }
        result_data["timing"] = {
            "engine_call": "scipy.optimize.brute",
            "setup_seconds": engine_started - setup_started,
            "engine_seconds": engine_finished - engine_started,
            "postprocess_seconds": perf_counter() - engine_finished,
        }
        return result_data

    ###############
    # DESCRIPTION #
    ###############

    def _description(self):
        return "Formulate as a brute-force search and solve using SciPy."

class SimulatedAnnealingAlgorithm(SingleTreeAlgorithm):
    """Simulated Annealing algorithm for single tree optimisation."""

    def __init__(self):
        super().__init__(name="Simulated Annealing")

        self.maxiter = 1000
        self.initial_temp = 5230.0
        self.restart_temp_ratio = 2e-05
        self.visit = 2.62
        self.accept = -5.0
        self.maxfun = 1e7
        self.random_seed = None
        self.progress_callback = None

    #########
    # SOLVE #
    #########

    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:
        """Simulated annealing."""

        setup_started = perf_counter()
        weighted_coefficients = _compute_weighted_coefficients(
            dataset, objectives, obj_weights, sense
        )

        f = partial(
            _weighted_objective,
            weighted_coefficients=weighted_coefficients,
        )

        coefficients, lower_bounds, upper_bounds = (
            self._constraint_system(dataset, constraints)
        )
        tolerance = 1e-12

        def binary_violation(x: np.ndarray) -> float:
            return _constraint_violation(
                x, coefficients, lower_bounds, upper_bounds
            )

        # The objective lies in [-objective_bound, objective_bound]. This scale
        # guarantees that any feasible point is preferred to any infeasible one,
        # while the violation term still guides the search toward feasibility.
        penalty_scale = _constraint_penalty_scale(weighted_coefficients)

        def annealing_objective(x: np.ndarray) -> float:
            binary_x = np.rint(x).astype(np.int8)
            violation = binary_violation(binary_x)
            value = f(binary_x)
            if violation > tolerance:
                value += penalty_scale * (1 + violation)
            return float(value)

        engine_started = perf_counter()
        result = scipy.optimize.dual_annealing(
            annealing_objective,
            bounds=[(0, 1)] * dataset._n_trees,
            maxiter=self.maxiter,
            initial_temp=self.initial_temp,
            restart_temp_ratio=self.restart_temp_ratio,
            visit=self.visit,
            accept=self.accept,
            maxfun=self.maxfun,
            rng=self.random_seed,
            no_local_search=True,
            callback=self.progress_callback,
        )
        engine_finished = perf_counter()

        best_x = np.rint(result.x).astype(np.int8)
        constraint_violation = binary_violation(best_x)

        result_data = {
            "status": result.message,
            "decisions": np.flatnonzero(best_x).tolist(),
            "constraint_violation": constraint_violation,
            "is_feasible": constraint_violation <= tolerance,
        }
        result_data["timing"] = {
            "engine_call": "scipy.optimize.dual_annealing",
            "setup_seconds": engine_started - setup_started,
            "engine_seconds": engine_finished - engine_started,
            "postprocess_seconds": perf_counter() - engine_finished,
        }
        return result_data

    def _description(self):
        return "Formulate as simulated annealing and solve using SciPy."

class GeneticAlgorithm(SingleTreeAlgorithm):
    """PyGAD genetic algorithm for binary single-tree optimisation."""

    def __init__(self):
        super().__init__(name="Genetic Algorithm")

        self.num_generations = 1000
        self.sol_per_pop = 50
        self.num_parents_mating = 10
        self.parent_selection_type = "sss"
        self.keep_elitism = 1
        self.crossover_type = "single_point"
        self.mutation_type = "random"
        self.mutation_probability = 0.05
        self.random_seed = None
        self.progress_callback = None

    def _solve(self, dataset: SingleTreeDataset, objectives: list[Objective],
        constraints: list[Constraint], obj_weights: list[float],
        sense: list[Literal["max", "min"]]) -> dict:
        """Evolve binary selections and penalise linear constraint violations."""

        setup_started = perf_counter()
        weighted_coefficients = _compute_weighted_coefficients(
            dataset, objectives, obj_weights, sense
        )
        
        coefficients, lower_bounds, upper_bounds = (
            self._constraint_system(dataset, constraints)
        )
        tolerance = 1e-12

        def constraint_violation(x: np.ndarray) -> float:
            return _constraint_violation(
                x, coefficients, lower_bounds, upper_bounds
            )

        # so that feasible points are always preferred to infeasible ones
        penalty_scale = _constraint_penalty_scale(weighted_coefficients)

        best_feasible_x = None
        best_feasible_fitness = -np.inf

        def fitness_func(ga_instance, solution, solution_idx):
            nonlocal best_feasible_x, best_feasible_fitness

            binary_x = np.asarray(solution, dtype=np.int8)
            if binary_x.ndim == 2:
                objective_fitness = binary_x @ weighted_coefficients
                violations = (
                    np.zeros(binary_x.shape[0])
                    if coefficients.shape[0] == 0
                    else np.asarray(_aggregate_constraint_violation(
                        binary_x.T, coefficients, lower_bounds, upper_bounds
                    )).ravel()
                )
                feasible = violations <= tolerance
                if np.any(feasible):
                    feasible_indices = np.flatnonzero(feasible)
                    batch_best_index = feasible_indices[
                        np.argmax(objective_fitness[feasible])
                    ]
                    batch_best_fitness = float(
                        objective_fitness[batch_best_index]
                    )
                    if batch_best_fitness > best_feasible_fitness:
                        best_feasible_fitness = batch_best_fitness
                        best_feasible_x = binary_x[batch_best_index].copy()
                return np.where(
                    feasible,
                    objective_fitness,
                    objective_fitness - penalty_scale * (1 + violations),
                ).tolist()

            objective_fitness = float(weighted_coefficients @ binary_x)
            violation = constraint_violation(binary_x)

            if violation <= tolerance:
                if objective_fitness > best_feasible_fitness:
                    best_feasible_fitness = objective_fitness
                    best_feasible_x = binary_x.copy()
                return objective_fitness

            return objective_fitness - penalty_scale * (1 + violation)

        def on_generation(ga_instance):
            if self.progress_callback is not None:
                return self.progress_callback(ga_instance)
            return None

        ga = pygad.GA(
            num_generations=int(self.num_generations),
            sol_per_pop=int(self.sol_per_pop),
            num_parents_mating=min(
                int(self.num_parents_mating), int(self.sol_per_pop)
            ),
            num_genes=dataset._n_trees,
            fitness_func=fitness_func,
            gene_space=[0, 1],
            gene_type=int,
            parent_selection_type=self.parent_selection_type,
            keep_elitism=int(self.keep_elitism),
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_probability=float(self.mutation_probability),
            random_seed=self.random_seed,
            on_generation=on_generation,
            fitness_batch_size=int(self.sol_per_pop),
        )
        engine_started = perf_counter()
        ga.run()
        engine_finished = perf_counter()

        if best_feasible_x is not None:
            best_x = best_feasible_x
        else:
            best_solution, _, _ = ga.best_solution()
            best_x = np.asarray(best_solution, dtype=np.int8)

        violation = constraint_violation(best_x)
        is_feasible = violation <= tolerance

        result_data = {
            "status": (
                "Feasible solution found by genetic algorithm."
                if is_feasible
                else "No feasible solution found by genetic algorithm."
            ),
            "decisions": np.flatnonzero(best_x).tolist(),
            "constraint_violation": violation,
            "is_feasible": is_feasible,
        }
        result_data["timing"] = {
            "engine_call": "pygad.GA.run",
            "setup_seconds": engine_started - setup_started,
            "engine_seconds": engine_finished - engine_started,
            "postprocess_seconds": perf_counter() - engine_finished,
        }
        return result_data

    def _description(self):
        return "Evolve binary selections using PyGAD with constraint penalties."
