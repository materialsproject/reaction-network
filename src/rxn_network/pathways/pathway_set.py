"""
Implements a class for conveniently and efficiently storing sets of ComputedReaction
objects which share entries.
"""

from functools import lru_cache
from typing import List, Optional, Union, Set, Iterable

import numpy as np
from monty.json import MSONable
from pymatgen.core import Element

from pymatgen.entries.computed_entries import ComputedEntry
from rxn_network.core.cost_function import CostFunction
from rxn_network.pathways import BasicPathway, BalancedPathway
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet


class PathwaySet(MSONable):
    """
    A lightweight class for storing large sets of Pathway objects.
    Automatically represents a set of pathways as a (non-rectangular) 2d array of
    indices corresponding to reactions within a reaction set.
    """

    def __init__(
        self,
        reaction_set: ReactionSet,
        indices: Union[np.ndarray, List[List[int]]],
        coefficients: List[float],
        costs: Optional[List[float]] = None,
    ):
        """
        Args:
            reaction_set: The reaction set containing all reactions in the pathways.
            indices: A list of lists of indices corresponding to reactions in the
                reaction set
            coefficients: A list of coefficients representing the multiplicities (how
                much of) each reaction in the pathway.
            costs: A list of costs for each pathway.
        """
        self.reaction_set = reaction_set
        self.indices = indices
        self.coefficients = coefficients
        self.costs = costs

        self._rxns = self.reaction_set.get_rxns()

    @lru_cache(1)
    def get_paths(
        self,
    ) -> List[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns list of BalancedPathway objects represented by the PathwaySet. Cached
        for efficiency.
        """
        paths = []
        for indices, coefficients, costs in zip(
            self.indices,
            self.coefficients,
            self.costs,
        ):
            reactions = [self._rxns[i] for i in indices]
            if coefficients is not None:
                path = BalancedPathway(
                    reactions=reactions, coefficients=coefficients, costs=costs
                )

            else:
                path = BasicPathway(reactions=reactions, costs=costs)

            paths.append(path)

        return paths

    @classmethod
    def from_paths(
        cls,
        paths: List[Union[ComputedReaction, OpenComputedReaction]],
    ) -> "PathwaySet":
        """
        Initiate a PathwaySet object from a list of paths. Including a list of
        unique entries saves some computation time.

        Args:
            paths: List of Pathway objects
        """
        indices, coefficients, costs = [], [], []

        reaction_set = cls._get_reaction_set(paths)
        rxns = reaction_set.get_rxns()

        for path in paths:
            indices.append([rxns.index(r) for r in path.reactions])
            coefficients.append(getattr(path, "coefficients", None))
            costs.append(path.costs)

        return cls(
            reaction_set=reaction_set,
            indices=indices,
            coefficients=coefficients,
            costs=costs,
        )

    @staticmethod
    def _get_reaction_set(
        paths: List[Union[ComputedReaction, OpenComputedReaction]],
    ) -> List[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns a reaction set built from a list of paths.

        Args:
            paths: List of Pathway objects
        """
        return ReactionSet.from_rxns([rxn for path in paths for rxn in path.reactions])

    def __iter__(self):
        """
        Iterates over the PathwaySet.
        """
        return iter(self.get_paths())
