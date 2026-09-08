from SingleTreeDataset import SingleTreeDataset
from abc import ABC, abstractmethod
from typing import Literal
from DecisionScope import DecisionScope, coerce_decision_scope

class Objective(ABC):
    """Base class for objective functions."""
    
    def __init__(self, name: str,
        sense: Literal["max", "min"] = "max",
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        
        self.dataset = None
        self.name = name
        self.sense = sense
        self.applies_to = applies_to
        self.type = None

    @property
    def applies_to(self) -> DecisionScope:
        return self._applies_to

    @applies_to.setter
    def applies_to(self, value: DecisionScope | str) -> None:
        self._applies_to = coerce_decision_scope(value)

    def bind(self, dataset: SingleTreeDataset) -> "Objective":
        """Bind this reusable objective definition to an inventory dataset."""
        self.dataset = dataset
        return self

    ################
    # CALCULATION #
    ################

    @abstractmethod
    def calculate(self, decisions):
        """Calculate objective value from a positional binary decision vector."""
        pass

    ###############
    # DESCRIPTION #
    ###############

    @abstractmethod
    def _description(self):
        """Description of the objective."""
        pass

    def describe(self):
        """Print description of the objective."""
        print(f"Objective: {self.name}\n")
        print(self._description())

class SumProductObjective(Objective):
    """Maximise sum-product of a numerical column and decisions."""
    
    def __init__(self, col_name: str, name: str,
        sense: Literal["max", "min"] = "max",
        applies_to: DecisionScope | str = DecisionScope.SELECT):
        self.column_key = col_name
        self.col_name = None
        super().__init__(name=name, sense=sense, applies_to=applies_to)
        self.type = 'sum_product'

    def bind(self, dataset: SingleTreeDataset) -> "SumProductObjective":
        super().bind(dataset)
        self.col_name = dataset._get_column_name(self.column_key)
        return self

    ###############
    # CALCULATION #
    ###############

    def calculate(self, decisions) -> float:
        """Calculate sum-product of a numerical column and decisions."""
        mask = self.dataset.decision_mask(decisions, self.applies_to)
        return float(self.dataset.data[self.col_name].to_numpy() @ mask)
    
    ###############
    # DESCRIPTION #
    ###############
    
    def _description(self):
        return f"""
        Maximise the sum-product of the '{self.col_name}' column and the decisions.

        Requires a numerical column in the dataset corresponding to '{self.col_name}'.
        """

class SocialStatusObjective(SumProductObjective):
    """Maximise social status of Z-trees."""
    
    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='social_status',
            name="Maximising social status", sense="max", applies_to=applies_to
        )

    ###############
    # DESCRIPTION #
    ###############
    
    def _description(self):
        return """
        Maximise the social status of selected trees.

        Requires a numerical, categorical column in the dataset corresponding to social status.
        Higher values indicate higher social status (e.g. 1= suppressed, ..., 5= dominant).
        """


class BranchlessObjective(SumProductObjective):
    """Maximise branchless trees among Z-trees."""
    
    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='pruned',
            name="Maximising branchless trees", sense="max", applies_to=applies_to
        )

    ###############
    # DESCRIPTION #
    ###############
    
    def _description(self):
        return """
        Maximise the proportion of branchless trees among selected trees.

        Requires a binary column in the dataset indicating whether the tree was pruned (1) or not (0).
        """

class QualityObjective(SumProductObjective):
    """Maximise wood quality among Z-trees."""
    
    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='quality',
            name="Maximising wood quality", sense="max", applies_to=applies_to
        )
    
    ###############
    # DESCRIPTION #
    ###############
    
    def _description(self):
        return """
        Maximise the total quality of selected trees.

        Requires a numerical column in the dataset corresponding to tree quality.
        Higher values indicate higher quality (e.g., 1=low, ..., 5=high).
        """

class MinDBHObjective(SumProductObjective):
    """Minimise DBH among Z-trees."""
    
    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='dbh', name="Minimising DBH", sense="min",
            applies_to=applies_to
        )
    
    ###############
    # DESCRIPTION #
    ###############
    
    def _description(self):
        return """
        Minimise the diameter at breast height (DBH) of selected trees.

        Requires a numerical column in the dataset corresponding to tree DBH.
        Higher values indicate larger trees.
        """


class MinVolumeObjective(SumProductObjective):
    """Minimise the volume of trees selected for thinning."""

    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='volume', name="Minimising selected-tree volume", sense="min",
            applies_to=applies_to
        )

    def _description(self):
        return """
        Minimise the total volume of selected trees.

        Requires a numerical volume column. Lower-volume trees are preferred.
        """


class MaxVolumeObjective(SumProductObjective):
    """Maximise the volume of trees selected for thinning."""

    def __init__(self, applies_to: DecisionScope | str = DecisionScope.SELECT):
        super().__init__(
            col_name='volume', name="Maximising selected-tree volume", sense="max",
            applies_to=applies_to
        )

    def _description(self):
        return """
        Maximise the total volume of selected trees.

        Requires a numerical volume column. Higher-volume trees are preferred.
        """
