"""
Implements a class for conveniently and efficiently storing sets of ComputedReaction
objects which share entries.
"""
from collections import OrderedDict
from functools import lru_cache
from itertools import combinations, groupby
from typing import Any, Collection, Dict, Iterable, List, Optional, Set, Union

import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction


class ReactionSet(MSONable):
    """
    A lightweight class for storing large sets of ComputedReaction objects.
    Automatically represents a set of reactions as an array of coefficients with
    a second array linking to a corresponding list of shared entries. This is useful for
    dumping large amounts of reaction data to a database.

    Note: this is not a true "set"; there is the option for filtering duplicates but it
        is not explicitly required.
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
            open_elem: Open element, e.g., "O"
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

    def get_rxns(
        self,
    ) -> Iterable[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Returns list of ComputedReaction objects or OpenComputedReaction objects (when
        open element and chempot are specified) for the reaction set.
        """
        return self._get_rxns_by_indices(idxs=range(len(self.coeffs)))

    @classmethod
    def from_rxns(
        cls,
        rxns: Collection[Union[ComputedReaction, OpenComputedReaction]],
        entries: Optional[Collection[ComputedEntry]] = None,
        open_elem: Optional[Union[str, Element]] = None,
        chempot: float = 0.0,
        filter_duplicates: bool = False,
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

        all_entry_indices: Dict[str, ComputedEntry] = {}
        indices, coeffs, data = [], [], []

        for rxn in rxns:
            rxn_indices = []
            for e in rxn.entries:
                idx = all_entry_indices.get(e.composition.reduced_formula)
                if idx is None:
                    idx = entries.index(e)
                    all_entry_indices[e.composition.reduced_formula] = idx
                rxn_indices.append(idx)

            indices.append(rxn_indices)
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

        rxn_set = cls(
            entries=entries,
            indices=indices,
            coeffs=coeffs,
            open_elem=open_elem,
            chempot=chempot,
            all_data=data,
        )

        if filter_duplicates:
            rxn_set = rxn_set.filter_duplicates()

        return rxn_set

    @lru_cache(maxsize=1)
    def to_dataframe(
        self,
        cost_function: CostFunction,
        target: Optional[Composition] = None,
        calculate_uncertainties=False,
        calculate_separable=False,
    ) -> DataFrame:
        """
        Make a dataframe of reactions from a ReactionSet object.

        Args:
            cost_function: Cost function to use for evaluating reaction costs
            target: Optional target composition (used to determine added elements)
            calculate_uncertainties: Whether to calculate uncertainties (dE column)
            calculate_separable: Whether to calculate if the reaction products are
                separable (see ComputedReaction.is_separable)

        Returns:
            Pandas DataFrame with columns:
                rxn: Reaction object
                energy: reaction energy in eV/atom
                dE (optional): uncertainty in reaction energy in eV/atom
                added_elems (optional): List of added elements
                separable (optional): whether reaction products are separable
                cost: Cost of reaction
                other: any other data associated with reaction

        """

        data: Dict[str, Any] = OrderedDict({k: [] for k in ["rxn", "energy"]})
        attrs = []

        calculate_e_above_hulls = False
        determine_theoretical = False

        for r in self.get_rxns():
            attrs = list(r.data.keys())  # get extra attributes from first reaction
            entry_data = r.entries[0].data
            if "e_above_hull" in entry_data:
                calculate_e_above_hulls = True
                data["max_e_hull_reactants"] = []
                data["max_e_hull_products"] = []
            if "icsd_ids" in entry_data or "theoretical" in entry_data:
                determine_theoretical = True
                data["num_theoretical_reactants"] = []
                data["num_theoretical_products"] = []

            break

        target = Composition(target) if target else None

        if "num_constraints" in attrs:
            attrs.remove("num_constraints")
        if calculate_uncertainties:
            data["dE"] = []
        if target:
            data["added_elems"] = []
            if calculate_separable:
                data["separable"] = []

        data["max_num_precursor_elems"] = []

        data.update({k: [] for k in attrs + ["cost"]})

        for rxn in self.get_rxns():
            data["rxn"].append(rxn)
            data["energy"].append(rxn.energy_per_atom)
            if calculate_uncertainties:
                data["dE"].append(rxn.energy_uncertainty_per_atom)
            if target:
                data["added_elems"].append(self._get_added_elems(rxn, target))
                if calculate_separable:
                    data["separable"].append(rxn.is_separable(target))

            if calculate_e_above_hulls:
                data["max_e_hull_reactants"].append(
                    max(e.data.get("e_above_hull", 0.0) for e in rxn.reactant_entries)
                )
                data["max_e_hull_products"].append(
                    max(e.data.get("e_above_hull", 0.0) for e in rxn.product_entries)
                )
            if determine_theoretical:
                data["num_theoretical_reactants"].append(
                    sum(bool(not e.is_experimental) for e in rxn.reactant_entries)
                )
                data["num_theoretical_products"].append(
                    sum(bool(not e.is_experimental) for e in rxn.product_entries)
                )

            data["max_num_precursor_elems"].append(
                max(len(precursor.elements) for precursor in rxn.reactants)
            )

            for attr in attrs:
                data[attr].append(rxn.data.get(attr))

            data["cost"].append(cost_function.evaluate(rxn))

        df = DataFrame(data).sort_values("cost").reset_index(drop=True)
        return df

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

    def add_rxns(self, rxns):
        """
        Return a new ReactionSet with the reactions added.

        Warning: all new reactions must only have entires contained in the entries of
        the current reaction set.
        """
        new_indices, new_coeffs, new_data = [], [], []
        for rxn in rxns:
            new_indices.append([self.entries.index(e) for e in rxn.entries])
            new_coeffs.append(list(rxn.coefficients))
            new_data.append(rxn.data)

        return ReactionSet(
            self.entries,
            self.indices + new_indices,
            self.coeffs + new_coeffs,
            self.open_elem,
            self.chempot,
            self.all_data + new_data,
        )

    def add_rxn_set(self, rxn_set):
        """Adds a new reaction set to current reaction set.

        Warning: new reaction set must have the same entries as the current reaction
        set.

        """
        if self.entries != rxn_set.entries:
            raise ValueError(
                "Reaction sets must have identical entries property to add."
            )
        open_elem = self.open_elem
        chempot = self.chempot

        indices = self.indices + rxn_set.indices
        coeffs = self.coeffs + rxn_set.coeffs
        all_data = self.all_data + rxn_set.all_data

        return ReactionSet(self.entries, indices, coeffs, open_elem, chempot, all_data)

    def get_rxns_by_reactants(self, reactants: List[str]):
        """
        Return a list of reactions with the given reactants.
        """
        idxs = []
        reactants = [Composition(r).reduced_formula for r in reactants]

        reactant_indices = {
            idx
            for idx, e in enumerate(self.entries)
            if e.composition.reduced_formula in reactants
        }

        if not reactant_indices:
            return []

        for idx, (coeffs, indices) in enumerate(zip(self.coeffs, self.indices)):
            r_indices = {i for c, i in zip(coeffs, indices) if c < 1e-12}
            if r_indices.issubset(reactant_indices):
                idxs.append(idx)

        return self._get_rxns_by_indices(idxs)

    def get_rxns_by_product(self, product: str):
        """
        Return a list of reactions which contain the given product formula.
        """
        idxs = []
        product = Composition(product).reduced_formula

        product_index = None
        for idx, e in enumerate(self.entries):
            if e.composition.reduced_formula == product:
                product_index = idx
                break

        if not product_index:
            return []

        for idx, (coeffs, indices) in enumerate(zip(self.coeffs, self.indices)):
            p_indices = {i for c, i in zip(coeffs, indices) if c > -1e-12}
            if product_index in p_indices:
                idxs.append(idx)

        return self._get_rxns_by_indices(idxs)

    def filter_duplicates(self):
        """
        Return a new ReactionSet object with duplicate reactions removed
        """
        indices_to_remove = set()
        if len(self.coeffs) == 0:
            return self

        # groupby only works with pre-sorted arrays
        sorted_coeffs, sorted_idxs, sorted_indices = zip(
            *list(
                sorted(
                    zip(self.coeffs, range(len(self.indices)), self.indices),
                    key=lambda x: sorted(x[2]),
                )
            )
        )

        for i, group in groupby(
            zip(sorted_coeffs, sorted_idxs, sorted_indices),
            key=lambda i: sorted(i[2]),
        ):
            coeffs_group, idx_group, indices_group = zip(*group)
            if len(idx_group) > 1:
                for (_, coeffs1, indices1), (
                    idx2,
                    coeffs2,
                    indices2,
                ) in combinations(zip(idx_group, coeffs_group, indices_group), 2):
                    if idx2 in indices_to_remove:
                        continue

                    coeffs2_sorted = [coeffs2[indices2.index(i)] for i in indices1]

                    ratios = np.array(coeffs1) / np.array(coeffs2_sorted)
                    if (
                        ratios <= 1e-8
                    ).any():  # do not remove any reaction with negative ratio
                        continue

                    if np.isclose(ratios[0], ratios).all():
                        indices_to_remove.add(idx2)

        new_indices = []
        new_coeffs = []
        new_all_data = []

        for idx in set(range(len(self))) - indices_to_remove:
            new_indices.append(self.indices[idx])
            new_coeffs.append(self.coeffs[idx])
            new_all_data.append(self.all_data[idx])

        return ReactionSet(
            self.entries,
            new_indices,
            new_coeffs,
            self.open_elem,
            self.chempot,
            new_all_data,
        )

    def _get_rxns_by_indices(
        self, idxs: Union[List[int], range]
    ) -> Iterable[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Return a list of reactions with the given indices.
        """
        if idxs == range(len(self.coeffs)):
            indices_slice = self.indices
            coeffs_slice = self.coeffs
            data_slice = self.all_data
        else:
            indices_slice = []
            coeffs_slice = []
            data_slice = []

            for idx in idxs:
                indices_slice.append(self.indices[idx])
                coeffs_slice.append(self.coeffs[idx])
                data_slice.append(self.all_data[idx])

        for indices, coeffs, data in zip(indices_slice, coeffs_slice, data_slice):
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

            yield rxn

    @staticmethod
    def _get_added_elems(
        rxn: Union[ComputedReaction, OpenComputedReaction], target: Composition
    ) -> str:
        """
        Get list of added elements for a reaction.

        Args:
            rxn: Reaction object
            target: target composition (used to determine added elements)

        """
        target = Composition(target)
        chemsys_prop = (
            "total_chemical_system"
            if rxn.__class__.__name__ == "OpenComputedReaction"
            else "chemical_system"
        )

        added_elems = set(getattr(rxn, chemsys_prop).split("-")) - set(
            target.chemical_system.split("-")
        )
        added_elems_str = "-".join(sorted(list(added_elems)))

        return added_elems_str

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

    def __len__(self):
        """
        Return length of reactions stored in the set.
        """
        return len(self.coeffs)
