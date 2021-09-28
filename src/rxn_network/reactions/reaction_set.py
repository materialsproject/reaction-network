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


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    Automatically represents a set of reactions as an array of coefficients with
    a second array linking to a corresponding list of shared entries.
    """

    def __init__(
        self,
        entries: List[ComputedEntry],
        indices: Union[np.ndarray, List[List[int]]],
        coeffs: Union[np.ndarray, List[List[float]]],
        all_data: Optional[List] = None,
    ):
        """
        Args:
            entries: List of ComputedEntry objects shared by reactions
            indices: Array indexing the entry list; gets entries used by each
                reaction object
            coeffs: Array of all reaction coefficients
            all_data: Optional list of data for each reaction
        """
        self.entries = entries
        self.indices = indices
        self.coeffs = coeffs
        self.all_data = all_data if all_data else []

    @lru_cache(1)
    def get_rxns(
        self,
        open_elem: Optional[str] = None,
        chempot: float = 0.0,
    ) -> List[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns list of ComputedReaction objects or OpenComputedReaction objects (when
        open element and chempot are specified) for the reaction set.

        Args:
            open_elem: Open element, e.g. "O2"
            chempot: Chemical potential (mu) of open element in equation: Phi = G - mu*N
        """
        rxns = []

        chempots = None

        if open_elem:
            chempots = {Element(open_elem): chempot}

        for indices, coeffs, data in zip(self.indices, self.coeffs, self.all_data):
            entries = [self.entries[i] for i in indices]
            if chempots:
                rxn = OpenComputedReaction(
                    entries=entries, coefficients=coeffs, data=data, chempots=chempots
                )
            else:
                rxn = ComputedReaction(entries=entries, coefficients=coeffs, data=data)
            rxns.append(rxn)
        return rxns

    def calculate_costs(
        self,
        cf: CostFunction,
        open_elem: Optional[Union[str, Element]] = None,
        chempot: Optional[float] = 0,
    ) -> List[float]:
        """
        Evaluate a cost function on an acquired set of reactions.

        Args:
            cf: CostFunction object, e.g. Softplus()
            open_elem: Open element, e.g. "O2"
            chempot: Chemical potential (mu) of open element in equation: Phi = G - mu*N
        """
        return [
            cf.evaluate(rxn)
            for rxn in self.get_rxns(open_elem=open_elem, chempot=chempot)
        ]

    @classmethod
    def from_rxns(
        cls,
        rxns: List[Union[ComputedReaction, OpenComputedReaction]],
        entries: Optional[Iterable[ComputedEntry]] = None,
    ) -> "ReactionSet":
        """
        Initiate a ReactionSet object from a list of reactions. Including a list of
        unique entries saves some computation time.

        Args:
            rxns: List of ComputedReaction-like objects.
            entries: Optional list of ComputedEntry objects
        """
        if not entries:
            entries = cls._get_unique_entries(rxns)

        entries = sorted(list(set(entries)), key=lambda r: r.composition)

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

    @staticmethod
    def _get_unique_entries(rxns: List[ComputedReaction]) -> Set[ComputedEntry]:
        """Return only unique entries from reactions"""
        entries = set()
        for r in rxns:
            entries.update(r.entries)
        return entries
