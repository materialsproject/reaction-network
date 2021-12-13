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
        reactions: ReactionSet,
        indices: Union[np.ndarray, List[List[int]]],
        coefficients: List[float],
        costs: Optional[List[float]] = None,
        all_data: Optional[List] = None,
    ):
        """
        Args:
            entries: List of ComputedEntry objects shared by reactions
            indices: 2d array indexing the list of reactions for each reaction
            all_data: 2d array  of data for each pathway
        """
        self.reaction_set = ReactionSet(reaction_set)
        self.indices = indices
        self.coefficients = coefficients
        self.costs = costs
        self.open_elem = open_elem
        self.chempot = chempot
        self.all_data = all_data if all_data else []

        self._rxns = self.reaction_set.get_rxns()

    @lru_cache(1)
    def get_paths(
        self,
    ) -> List[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns list of ComputedReaction objects or OpenComputedReaction objects (when
        open element and chempot are specified) for the reaction set.

        Args:
            open_elem: Open element, e.g. "O2"
            chempot: Chemical potential (mu) of open element in equation: Phi = G - mu*N
        """
        paths = []
        for indices, coeffs, costs, data in zip(
            self.indices, self.coefficients, self.costs, self.all_data
        ):
            rxns = [self._rxns[i] for i in indices]
            if chempots:
                rxn = OpenComputedReaction(
                    entries=entries, coefficients=coeffs, data=data, chempots=chempots
                )
            else:
                rxn = ComputedReaction(entries=entries, coefficients=coeffs, data=data)
            rxns.append(rxn)
        return rxns

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
        all_indices, all_coeffs, all_data = [], [], []
        for rxn in rxns:
            all_indices.append([entries.index(e) for e in rxn.entries])
            all_coeffs.append(list(rxn.coefficients))
            all_data.append(rxn.data)

        return cls(
            entries=entries,
            indices=all_indices,
            coeffs=all_coeffs,
            all_data=all_data,
        )
