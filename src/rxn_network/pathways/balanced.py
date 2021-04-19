" A balanced reaction pathway class"

from typing import List, Optional, Union

import numpy as np

from rxn_network.core import Pathway, Reaction
from rxn_network.pathways.basic import BasicPathway


class BalancedPathway(BasicPathway):
    """
    Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes cost of each
    reaction.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        coefficients: List[float],
        costs: Optional[List[float]] = None,
    ):
        """
        Args:
            rxns ([ComputedReaction]): list of ComputedReaction objects in pymatgen
                which occur along path.
            costs ([float]): list of corresponding costs for each reaction.
        """
        self.coefficients = coefficients
        super().__init__(reactions=reactions, costs=costs)

    def __eq__(self, other):
        if super().__eq__(other):
            return np.allclose(self.costs, other.costs)

        return False

    def __hash__(self):
        return hash(tuple(self.reactions, self.coefficients))

    @classmethod
    def balance(
        cls,
        pathway_sets: Union[List[Pathway], List[List[Reaction]]],
        net_reaction: Reaction,
    ):
        """
        Balances multiple reaction pathways to a net reaction
        """
        raise NotImplementedError(
            "Matt please implement using your numba optimized method"
        )
