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
        indices: Dict[int, np.ndarray],
        coeffs: Dict[int, np.ndarray],
        open_elem: Optional[Union[str, Element]] = None,
        chempot: float = 0.0,
        all_data: Optional[Dict[int, np.ndarray]] = None,
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
        self.all_data = all_data

        if all_data is None:
            all_data = {i: np.array([]) for i in self.indices}

        self.mu_dict = None

        if open_elem:
            self.mu_dict = {Element(open_elem): chempot}  # type: ignore

    def get_rxns(
        self,
    ) -> Iterable[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Generator for all ComputedReaction objects or OpenComputedReaction objects (when
        open element and chempot are specified) for the reaction set.
        """
        return self._get_rxns_by_indices(
            idxs={i: slice(0, len(c)) for i, c in self.indices.items()}
        )

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
        indices, coeffs, data = {}, {}, {}  # keys are reaction sizes

        for rxn in rxns:
            rxn_indices = []
            size = len(rxn.entries)

            for e in rxn.entries:
                idx = all_entry_indices.get(e)
                if idx is None:
                    idx = entries.index(e)
                    all_entry_indices[e] = idx

                rxn_indices.append(idx)

            if size not in indices:
                indices[size] = []
                coeffs[size] = []
                data[size] = []

            indices[size].append(rxn_indices)
            coeffs[size].append(list(rxn.coefficients))
            data[size].append(rxn.data)

        for size in indices:
            indices[size] = np.array(indices[size])
            coeffs[size] = np.array(coeffs[size])
            data[size] = np.array(data[size])

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
            np.concatenate((self.indices, np.array(new_indices))),
            np.concatenate((self.coeffs, np.array(new_coeffs))),
            self.open_elem,
            self.chempot,
            np.concatenate((self.all_data, np.array(new_data))),
        )

    def add_rxn_set(self, rxn_set: "ReactionSet") -> "ReactionSet":
        """Adds a new reaction set to current reaction set.

        Warning: new reaction set must have the same entries as the current reaction
        set.

        """
        if self.entries != rxn_set.entries:
            raise ValueError(
                "Reaction sets must have identical entries property to combine."
            )
        open_elem = self.open_elem
        chempot = self.chempot

        for size in rxn_set.indices:
            if size not in self.indices:
                self.indices[size] = rxn_set.indices[size]
                self.coeffs[size] = rxn_set.coeffs[size]
                self.all_data[size] = rxn_set.all_data[size]
            else:
                indices = np.concatenate((self.indices[size], rxn_set.indices[size]))
                coeffs = np.concatenate((self.coeffs[size], rxn_set.coeffs[size]))
                all_data = np.concatenate((self.all_data[size], rxn_set.all_data[size]))

        return ReactionSet(self.entries, indices, coeffs, open_elem, chempot, all_data)

    def get_rxns_by_reactants(
        self, reactants: List[str], return_set: bool = False
    ) -> Iterable[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Return a list of reactions with the given reactants.
        """
        reactants = [Composition(r).reduced_formula for r in reactants]

        reactant_indices = list(
            {
                idx
                for idx, e in enumerate(self.entries)
                if e.composition.reduced_formula in reactants
            }
        )

        if not reactant_indices:
            return []

        idxs: Dict[int, List[int]] = {}
        for size, indices in self.indices.items():
            idxs[size] = []
            contains_reactants = np.isin(indices, reactant_indices).sum(axis=1) == len(
                reactant_indices
            )
            for idx, coeffs, indices in zip(
                np.argwhere(contains_reactants).flatten(),
                self.coeffs[size][contains_reactants],
                self.indices[size][contains_reactants],
            ):
                r_indices = {i for c, i in zip(coeffs, indices) if c < 1e-12}
                if r_indices.issubset(reactant_indices):
                    idxs[size].append(idx)

        if return_set:
            return self._get_rxn_set_by_indices(idxs)

        return self._get_rxns_by_indices(idxs)

    def get_rxns_by_product(self, product: str, return_set: bool = False):
        """
        Return a list of reactions which contain the given product formula.
        """
        product = Composition(product).reduced_formula

        product_index = None
        for idx, e in enumerate(self.entries):
            if e.composition.reduced_formula == product:
                product_index = idx
                break

        if not product_index:
            return []

        idxs: Dict[int, np.ndarray] = {}
        for size, indices in self.indices.items():
            idxs[size] = []
            contains_product = np.isin(indices, product_index).any(axis=1)

            for idx, coeffs, indices in zip(
                np.argwhere(contains_product).flatten(),
                self.coeffs[size][contains_product],
                self.indices[size][contains_product],
            ):
                p_indices = {i for c, i in zip(coeffs, indices) if c > -1e-12}
                if product_index in p_indices:
                    idxs[size].append(idx)

        if return_set:
            return self._get_rxn_set_by_indices(idxs)

        return self._get_rxns_by_indices(idxs)

    def filter_duplicates(self, ensure_rxns: Optional[List[ComputedReaction]] = None):
        """
        Return a new ReactionSet object with duplicate reactions removed
        """

        def get_idxs_to_remove(selected, possible):
            """Looks for reactions with coeffs that are positive multiples
            of each other."""
            multiples = possible / selected
            return np.argwhere(
                np.apply_along_axis(
                    lambda r: np.isclose(abs(r[0]), r), axis=1, arr=multiples
                ).all(axis=1)
            ).flatten()

        ensure_rxn_data = {}
        filter_idxs = {size: [] for size in self.indices}

        for rxn in ensure_rxns or []:
            rxn_idxs = np.array([self.entries.index(e) for e in rxn.entries])
            rxn_coeffs = rxn.coefficients

            size = len(rxn_idxs)
            if size in ensure_rxn_data:
                ensure_rxn_data[size].append((rxn_idxs, rxn_coeffs))
            else:
                ensure_rxn_data[size] = [(rxn_idxs, rxn_coeffs)]

        duplicate_rxns = {size: [] for size in self.indices}

        for size in self.indices:
            rxn_idxs_to_remove = []
            rxn_idxs_to_keep = []

            if size in ensure_rxn_data:
                for rxn_idxs, rxn_coeffs in ensure_rxn_data[size]:
                    possible_duplicates = np.argwhere(
                        (self.indices[size] == rxn_idxs).all(axis=1)
                    ).flatten()
                    duplicates = np.argwhere(
                        np.isclose(
                            self.coeffs[size][possible_duplicates], rxn_coeffs
                        ).all(axis=1)
                    ).flatten()
                    rxn_idxs_to_keep.append(possible_duplicates[duplicates[0]])

            sorting_indices = np.argsort(self.indices[size], axis=1)

            indices = np.take_along_axis(self.indices[size], sorting_indices, axis=1)
            coeffs = np.take_along_axis(self.coeffs[size], sorting_indices, axis=1)

            _, unique_idxs, counts = np.unique(
                indices, return_index=True, return_counts=True, axis=0
            )
            idxs_to_check = unique_idxs[np.argwhere(counts > 1).flatten()]

            for idx in idxs_to_check:
                current_coeffs = coeffs[idx]
                current_indices = indices[idx]
                possible_duplicate_idxs = np.argwhere(
                    (indices == current_indices).all(axis=1)
                ).flatten()
                possible_duplicate_coeffs = coeffs[possible_duplicate_idxs]
                print(current_coeffs, possible_duplicate_coeffs)

                duplicate_idxs = possible_duplicate_idxs[
                    get_idxs_to_remove(current_coeffs, possible_duplicate_coeffs)
                ]
                print(duplicate_idxs)
                print("\n")
                if np.isin(rxn_idxs_to_keep, duplicate_idxs).any():
                    rxn_idxs_to_remove.extend(
                        [i for i in duplicate_idxs if i not in rxn_idxs_to_keep]
                    )
                elif len(duplicate_idxs) > 1:
                    rxn_idxs_to_remove.extend(duplicate_idxs[1:].tolist())

            filter_idxs[size] = [
                i
                for i in range(0, len(self.indices[size]))
                if i not in rxn_idxs_to_remove
            ]

            print(list(self._get_rxns_by_indices(idxs={size: rxn_idxs_to_remove})))

        return self._get_rxn_set_by_indices(filter_idxs)

    def _get_rxns_by_indices(
        self, idxs: Dict[int, np.ndarray]
    ) -> Iterable[Union[ComputedReaction, OpenComputedReaction]]:
        """
        Return a list of reactions with the given indices.
        """

        for size, idx_arr in idxs.items():
            if not idx_arr:
                idx_arr = slice(0, 0)
            elif not isinstance(idx_arr, np.ndarray) and not isinstance(idx_arr, slice):
                idx_arr = np.array(idx_arr)

            for indices, coeffs, data in zip(
                self.indices[size][idx_arr],
                self.coeffs[size][idx_arr],
                self.all_data[size][idx_arr],
            ):
                entries = [self.entries[i] for i in indices]
                if self.mu_dict:
                    rxn = OpenComputedReaction(
                        entries=entries,
                        coefficients=coeffs,
                        data=data,
                        chempots=self.mu_dict,
                    )
                else:
                    rxn = ComputedReaction(
                        entries=entries, coefficients=coeffs, data=data
                    )

                yield rxn

    def _get_rxn_set_by_indices(self, idxs: Dict[int, np.ndarray]) -> "ReactionSet":
        """
        Return a list of reactions with the given indices.
        """

        return ReactionSet(
            entries=self.entries,
            indices={
                size: self.indices[size][idx_arr] for size, idx_arr in idxs.items()
            },
            coeffs={size: self.coeffs[size][idx_arr] for size, idx_arr in idxs.items()},
            open_elem=self.open_elem,
            chempot=self.chempot,
            all_data={
                size: self.all_data[size][idx_arr] for size, idx_arr in idxs.items()
            },
        )

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
    def _get_entry_key(entry: ComputedEntry) -> str:
        """
        Get a unique key for an entry. Assumes that a formula and energy alone should be
        able to define a unique entry.

        Args:
            entry: Entry object

        """
        return f"{entry.composition.reduced_formula}_{round(entry.energy_per_atom, 4)}"

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
        return sum(len(i) for i in self.indices.values())
