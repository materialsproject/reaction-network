"""
Implements a class for conveniently and efficiently storing sets of ComputedReaction
objects which share entries.
"""
from functools import lru_cache
from typing import Collection, List, Optional, Set, Union

import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from pymatgen.core.composition import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core.cost_function import CostFunction
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    Automatically represents a set of reactions as an array of coefficients with
    a second array linking to a corresponding list of shared entries. This is useful for
    dumping large amounts of reaction data to a database.
    """

    def __init__(
        self,
        entries: List[ComputedEntry],
        indices: Union[np.ndarray, List[List[int]]],
        coeffs: Union[np.ndarray, List[List[float]]],
        open_elem: Optional[Union[str, Element]] = None,
        chempot: float = 0.0,
        all_data: Optional[List] = None,
    ):
        """
        Args:
            entries: List of ComputedEntry objects shared by reactions
            indices: Array indexing the entry list; gets entries used by each
                reaction object
            coeffs: Array of all reaction coefficients
            open_elem: Open element, e.g. "O2"
            chempot: Chemical potential (mu) of open element in equation: Phi = G - mu*N
            all_data: Optional list of data for each reaction
        """
        self.entries = entries
        self.indices = indices
        self.coeffs = coeffs
        self.open_elem = open_elem
        self.chempot = chempot
        self.all_data = all_data if all_data else []

        self.mu_dict = None
        if open_elem:
            self.mu_dict = {Element(open_elem): chempot}  # type: ignore

    @lru_cache(1)
    def get_rxns(
        self,
    ) -> List[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns list of ComputedReaction objects or OpenComputedReaction objects (when
        open element and chempot are specified) for the reaction set.
        """
        rxns = []

        for indices, coeffs, data in zip(self.indices, self.coeffs, self.all_data):
            entries = [self.entries[i] for i in indices]
            if self.mu_dict:
                rxn = OpenComputedReaction(
                    entries=entries,
                    coefficients=coeffs,
                    data=data,
                    chempots=self.mu_dict,
                )
            else:
                rxn = ComputedReaction(entries=entries, coefficients=coeffs, data=data)
            rxns.append(rxn)
        return rxns

    def calculate_costs(
        self,
        cf: CostFunction,
    ) -> List[float]:
        """
        Evaluate a cost function on an acquired set of reactions.

        Args:
            cf: CostFunction object, e.g. Softplus()
        """
        return [cf.evaluate(rxn) for rxn in self.get_rxns()]

    @classmethod
    def from_rxns(
        cls,
        rxns: Collection[Union[ComputedReaction, OpenComputedReaction]],
        entries: Optional[Collection[ComputedEntry]] = None,
        open_elem: Optional[Union[str, Element]] = None,
        chempot: float = 0.0,
    ) -> "ReactionSet":
        """
        Initiate a ReactionSet object from a list of reactions. Including a list of
        unique entries saves some computation time.

        Args:
            rxns: List of ComputedReaction-like objects.
            entries: Optional list of ComputedEntry objects
            open_elem: Open element, e.g. "O2"
            chempot: Chemical potential (mu) of open element in equation: Phi = G - mu*N
        """
        if not entries:
            entries = cls._get_unique_entries(rxns)

        entries = sorted(list(set(entries)), key=lambda r: r.composition)

        indices, coeffs, data = [], [], []
        for rxn in rxns:
            indices.append([entries.index(e) for e in rxn.entries])
            coeffs.append(list(rxn.coefficients))
            data.append(rxn.data)

        all_open_elems: Set[Element] = set()
        all_chempots: Set[float] = set()

        if (
            all(r.__class__.__name__ == "OpenComputedReaction" for r in rxns)
            and not open_elem
        ):
            for r in rxns:
                all_open_elems.update(r.chempots.keys())
                all_chempots.update(r.chempots.values())

            if len(all_chempots) == 1 and len(all_open_elems) == 1:
                chempot = all_chempots.pop()
                open_elem = all_open_elems.pop()

        return cls(
            entries=entries,
            indices=indices,
            coeffs=coeffs,
            open_elem=open_elem,
            chempot=chempot,
            all_data=data,
        )

    @lru_cache(1)
    def to_dataframe(
        self, cost_function: CostFunction, target: Optional[Composition] = None
    ) -> DataFrame:
        """
        Make a dataframe of reactions from a ReactionSet object.

        Args:
            cost_function: Cost function to use for evaluating reaction costs
            target: Optional target composition (used to determine added elements)

        Returns:
            Pandas DataFrame with columns:
                rxn: Reaction object
                energy: reaction energy in eV/atom
                distance: Distance in eV/atom
                added_elems: List of added elements
                cost: Cost of reaction

        """
        rxns = self.get_rxns()
        costs = [cost_function.evaluate(rxn) for rxn in rxns]

        if target:
            target = Composition(target)
            if rxns[0].__class__.__name__ == "OpenComputedReaction":
                added_elems = [
                    rxn.total_chemical_system != target.chemical_system for rxn in rxns
                ]
            else:
                added_elems = [
                    rxn.chemical_system != target.chemical_system for rxn in rxns
                ]
        else:
            added_elems = [None] * len(rxns)

        data = {
            "rxn": rxns,
            "energy": [rxn.energy_per_atom for rxn in rxns],
            "dE": [rxn.energy_uncertainty_per_atom for rxn in rxns],  # type: ignore
            "added_elems": added_elems,
        }

        attrs = list(rxns[0].data.keys())
        for attr in attrs:
            data.update({attr: [rxn.data.get(attr) for rxn in rxns]})

        data["cost"] = costs

        df = DataFrame(data).sort_values("cost").reset_index(drop=True)

        return df

    @staticmethod
    def _get_unique_entries(rxns: Collection[ComputedReaction]) -> Set[ComputedEntry]:
        """
        Return only unique entries from reactions
        """
        entries = set()
        for r in rxns:
            entries.update(r.entries)
        return entries

    def __iter__(self):
        """
        Iterate over the reactions in the set.
        """
        return iter(self.get_rxns())
