from typing import List
import numpy as np

from rxn_network.core import CostFunction
from rxn_network.reactions import ComputedReaction


class Softplus(CostFunction):
    def __init__(
        self,
        temp: float = 300,
        params: List[str] = ["energy_per_atom"],
        weights: List[float] = [1.0],
    ):

        self.temp = temp
        self.params = params
        self.weights = np.array(weights)

    def evaluate(self, rxn: ComputedReaction) -> float:
        """
        Calculates cost of reaction based on initialized weights/parameters.
        Args:
            rxn (ComputedReaction): Reaction object

        Returns:
            Cost
        """
        values = []
        for p in self.params:
            if rxn.data and p in rxn.data:
                value = rxn.data[p]
            elif hasattr(rxn, p):
                value = getattr(rxn, p)
            else:
                raise ValueError(f"Reaction is missing parameter {p}!")
            values.append(value)

        values = np.array(values)
        total = np.dot(values, self.weights)

        return self._softplus(total, self.temp)

    @staticmethod
    def _softplus(x, t):
        return np.log(1 + (273 / t) * np.exp(x))

    def __repr__(self):
        return (
            f"Softplus with parameters: "
            f"{' '.join([(f'{k} ({v})') for k, v in zip(self.params, self.weights)])}"
        )
