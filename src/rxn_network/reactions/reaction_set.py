"""
Implements a class for conveniently and efficiently storing sets of ComputedReaction
objects which share entries.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Collection, Iterable

import numpy as np
import ray
from monty.json import MSONable
from pandas import DataFrame
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core import Composition
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.utils.funcs import get_logger, grouper
from rxn_network.utils.ray import initialize_ray, to_iterator

if TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedEntry

    from rxn_network.costs.base import CostFunction

logger = get_logger(__name__)


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
        entries: list[ComputedEntry],
        indices: dict[int, np.ndarray],
        coeffs: dict[int, np.ndarray],
        open_elem: str | Element | None = None,
        chempot: float = 0.0,
        all_data: dict[int, np.ndarray] | None = None,
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
        if all_data is None:
            self.all_data = {int(i): np.array([]) for i in self.indices}
        else:
            self.all_data = all_data

        if not all(isinstance(k, int) for k in self.indices.keys()) or not all(
            isinstance(v, np.ndarray) for v in self.indices.values()
        ):
            self.indices = {
                int(size): np.array(arr) for size, arr in self.indices.items()
            }

        if not all(isinstance(k, int) for k in self.coeffs.keys()) or not all(
            isinstance(v, np.ndarray) for v in self.coeffs.values()
        ):
            self.coeffs = {
                int(size): np.array(arr) for size, arr in self.coeffs.items()
            }

        if not all(isinstance(k, int) for k in self.all_data.keys()) or not all(
            isinstance(v, np.ndarray) for v in self.all_data.values()
        ):
            self.all_data = {
                int(size): np.array(arr) for size, arr in self.all_data.items()
            }

        self.open_elem = open_elem
        self.chempot = chempot

        self.mu_dict = None

        if open_elem:
            self.mu_dict = {Element(open_elem): chempot}  # type: ignore

    def get_rxns(
        self,
    ) -> Iterable[ComputedReaction | OpenComputedReaction]:
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
        rxns: Collection[ComputedReaction | OpenComputedReaction],
        entries: Collection[ComputedEntry] | None = None,
        open_elem: str | Element | None = None,
        chempot: float = 0.0,
        filter_duplicates: bool = False,
    ) -> ReactionSet:
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

        # need to index by unique ID in case mixing different temperatures
        all_entry_indices = {}
        for idx, entry in enumerate(entries):
            if hasattr(entry, "unique_id"):
                all_entry_indices[entry.unique_id] = idx
            else:
                all_entry_indices[entry.entry_id] = idx

        # group by reaction size to ensure rectangular matrices
        indices, coeffs, data = {}, {}, {}  # type: ignore

        for rxn in rxns:
            size = len(rxn.entries)

            rxn_indices = []
            for e in rxn.entries:
                if hasattr(e, "unique_id"):
                    rxn_indices.append(all_entry_indices[e.unique_id])
                else:
                    rxn_indices.append(all_entry_indices[e.entry_id])

            if size not in indices:
                indices[size] = []
                coeffs[size] = []
                data[size] = []

            indices[size].append(rxn_indices)
            coeffs[size].append(rxn.coefficients)
            data[size].append(rxn.data)

        for size in indices:
            indices[size] = np.array(indices[size])
            coeffs[size] = np.array(coeffs[size])
            data[size] = np.array(data[size])

        all_open_elems: set[Element] = set()
        all_chempots: set[float] = set()

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
        target: Composition | None = None,
        calculate_uncertainties: bool = False,
        calculate_separable: bool = False,
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
        data: dict[str, Any] = OrderedDict({k: [] for k in ["rxn", "energy"]})
        attrs = []

        calculate_e_above_hulls = False
        determine_theoretical = False

        for r in self.get_rxns():
            attrs = list(r.data.keys())  # get extra attributes from first reaction
            entry_data = r.entries[0].data
            if entry_data.get("e_above_hull") is not None:
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
                    max(
                        e.data.get("e_above_hull", 0.0) or 0.0
                        for e in rxn.reactant_entries
                    )
                )
                data["max_e_hull_products"].append(
                    max(
                        e.data.get("e_above_hull", 0.0) or 0.0
                        for e in rxn.product_entries
                    )
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
    ) -> list[float]:
        """
        Evaluate a cost function on an acquired set of reactions.

        Args:
            cf: CostFunction object, e.g. Softplus()
        """
        return [cf.evaluate(rxn) for rxn in self.get_rxns()]

    def add_rxns(self, rxns: Collection[ComputedReaction | OpenComputedReaction]):
        """
        Return a new ReactionSet with the reactions added.

        Warning: all new reactions must only have entires contained in the entries of
        the current reaction set.
        """
        new_rxn_set = ReactionSet.from_rxns(rxns, entries=self.entries)

        return self.add_rxn_set(new_rxn_set)

    def add_rxn_set(self, rxn_set: ReactionSet) -> ReactionSet:
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

        new_indices = {}
        new_coeffs = {}
        new_all_data = {}

        for size in set(list(self.indices.keys()) + list(rxn_set.indices.keys())):
            if size in self.indices and size in rxn_set.indices:
                new_indices[size] = np.concatenate(
                    (self.indices[size], rxn_set.indices[size])
                )
                new_coeffs[size] = np.concatenate(
                    (self.coeffs[size], rxn_set.coeffs[size])
                )
                new_all_data[size] = np.concatenate(
                    (self.all_data[size], rxn_set.all_data[size])
                )
            elif size in self.indices:
                new_indices[size] = self.indices[size]
                new_coeffs[size] = self.coeffs[size]
                new_all_data[size] = self.all_data[size]
            elif size in rxn_set.indices:
                new_indices[size] = rxn_set.indices[size]
                new_coeffs[size] = rxn_set.coeffs[size]
                new_all_data[size] = rxn_set.all_data[size]

        return ReactionSet(
            self.entries, new_indices, new_coeffs, open_elem, chempot, new_all_data
        )

    def get_rxns_by_reactants(
        self, reactants: list[str], return_set: bool = False
    ) -> Iterable[ComputedReaction | OpenComputedReaction]:
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

        idxs = {}  # type: ignore
        for size, indices in self.indices.items():
            idxs[size] = []
            contains_reactants = np.isin(indices, reactant_indices).any(axis=1)
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

        Args:
            product: The product's formula
            return_set: Whether to return the identified reactions in the form of a
                ReactionSet object. Defaults to False.

        """
        product = Composition(product).reduced_formula

        product_index = None
        for idx, e in enumerate(self.entries):
            if e.composition.reduced_formula == product:
                product_index = idx
                break

        if not product_index:
            return []

        idxs = {}  # type: ignore
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

    def filter_duplicates(
        self,
        ensure_rxns: list[ComputedReaction | OpenComputedReaction] | None = None,
        parallelize: bool = True,
    ) -> ReactionSet:
        """
        Returns a new ReactionSet object with duplicate reactions removed.

        NOTE: Duplicate reactions include those that are multiples of each other. For
        example, if a reaction set contains both A + B -> C and 2A + 2B -> 2C, the
        second reaction will be removed.

        Args:
            ensure_rxns: An optional list of reactions to ensure are contained within
                the filtered set that is returned. This is important for some cases
                (e.g., pathfinding), where you expect a certain reaction object to be in
                the set.
            parallelize: Whether to parallelize duplicate checking with Ray. This can be
                a slow procedure otherwise. Defaults to True.
        """
        if parallelize:
            if not ray.is_initialized():
                initialize_ray()
            num_cpus = int(ray.cluster_resources()["CPU"]) - 1
            chunk_refs = []

        ensure_idxs = {size: [] for size in self.indices}  # type: ignore
        idxs_to_keep = {size: [] for size in self.indices}  # type: ignore

        # identify indices to keep
        for rxn in ensure_rxns or []:
            rxn_idxs = np.array([self.entries.index(e) for e in rxn.entries])
            rxn_coeffs = rxn.coefficients

            size = len(rxn_idxs)

            # filter by first column to save some time (full filter is long)
            quick_filter = np.argwhere(
                rxn_idxs[0] == self.indices[size][:, 0]
            ).flatten()

            for i, idxs, coeffs in zip(
                quick_filter,
                self.indices[size][quick_filter],
                self.coeffs[size][quick_filter],
            ):
                if (idxs == rxn_idxs).all() and (coeffs == rxn_coeffs).all():
                    ensure_idxs[size].append(i)
                    break
            else:
                raise ValueError(f"{rxn} not found in ReactionSet!")

        for size in sorted(self.indices.keys()):
            ensure_idxs_to_keep = ensure_idxs[size]

            # reordering columns in ascending entry idx
            column_sorting_indices = np.argsort(self.indices[size], axis=1)
            indices = np.take_along_axis(
                self.indices[size], column_sorting_indices, axis=1
            )
            coeffs = np.take_along_axis(
                self.coeffs[size], column_sorting_indices, axis=1
            )

            _, inverse_indices = np.unique(
                indices,
                axis=0,
                return_inverse=True,
            )

            keep = []
            group_indices = []

            for i in np.split(
                np.argsort(inverse_indices),
                np.cumsum(np.unique(inverse_indices, return_counts=True)[1])[:-1],
            ):  # this was suggested by ChatGPT
                if len(i) == 1:
                    keep.append(i[0])
                else:
                    group_indices.append(i)

            num_groups = len(group_indices)
            idxs_to_keep[size] = keep

            if num_groups == 0:
                continue

            if parallelize:
                chunk_size = (num_groups // num_cpus) + 1

                coeffs = ray.put(coeffs)

                for chunk in grouper(
                    group_indices,
                    chunk_size,
                    fillvalue=None,
                ):
                    chunk_refs.append(
                        _process_duplicates_ray.remote(
                            size, chunk, coeffs, ensure_idxs_to_keep
                        )
                    )
            else:
                idxs_to_keep[size].extend(
                    _process_duplicates(group_indices, coeffs, ensure_idxs_to_keep)
                )

        if parallelize:
            for result in tqdm(
                to_iterator(chunk_refs),
                total=len(chunk_refs),
                desc="Filtering duplicates",
            ):
                size, data = result
                idxs_to_keep[size].extend(data)

        return self._get_rxn_set_by_indices(idxs_to_keep)

    def set_chempot(self, open_el: str | Element | None, chempot: float) -> ReactionSet:
        """
        Returns a new ReactionSet containing the same reactions as this ReactionSet but
        with a grand potential change recalculated under the constraint defined by the
        provided open element and its chemical potential.

        Args:
            open_el: The element to be considered open.
            chempot: The open element's chemical potential (for use in energy change
                calculation)

        Returns:
            ReactionSet: A new ReactionSet containing reactions with the recalculated
                energies.
        """
        return ReactionSet(
            self.entries, self.indices, self.coeffs, open_el, chempot, self.all_data
        )

    def set_new_temperature(self, new_temp: float) -> ReactionSet:
        """
        Returns a new ReactionSet containing the same reactions as this ReactionSet but
        with a recalculated Gibb's/Grand potential change reflecting formation energies
        calculated at the provided temperature.

        Args:
            new_temp: The temperature for which new reaction energies should be
                calculated.

        Returns:
            The new ReactionSet containing the recalculated reactions.
        """
        new_rxns = []
        for rxn in self.get_rxns():
            new_rxns.append(rxn.get_new_temperature(new_temp))

        return ReactionSet.from_rxns(
            new_rxns, open_elem=self.open_elem, chempot=self.chempot
        )

    def _get_rxns_by_indices(
        self, idxs
    ) -> Iterable[ComputedReaction | OpenComputedReaction]:
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

    def _get_rxn_set_by_indices(self, idxs) -> ReactionSet:
        """
        Return a list of reactions with the given indices. This is the backbone of other
        reaction subset generation methods.
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
        rxn: ComputedReaction | OpenComputedReaction, target: Composition | str
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
    def _get_unique_entries(rxns: Collection[ComputedReaction]) -> set[ComputedEntry]:
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


def _get_idxs_to_keep(rows, ensure_idxs=None):
    """Looks for reactions with coeffs that are positive multiples of each other."""
    sorted_indices = np.argsort(np.abs(rows).sum(axis=1))  # sort by abs sum
    sorted_rows = rows[sorted_indices]
    if ensure_idxs is not None:
        ensure_idxs = set(ensure_idxs)

    to_keep, to_remove = [], []
    for idx, row in zip(sorted_indices, sorted_rows):
        if idx in to_remove:
            continue

        multiples = sorted_rows / row
        duplicates = np.argwhere(
            np.apply_along_axis(
                lambda r: np.isclose(abs(r[0]), r), axis=1, arr=multiples
            ).all(axis=1)
        ).flatten()

        group = sorted_indices[duplicates]
        if ensure_idxs:
            group_set = set(group)
            overlap = ensure_idxs & group_set

            if overlap:
                if not overlap.issubset(to_keep):
                    difference = group_set - ensure_idxs
                    to_keep.extend(list(overlap))
                    to_remove.extend(list(difference))
            else:
                to_keep.append(group[0])
                to_remove.extend(group[1:])
        else:
            to_keep.append(group[0])
            to_remove.extend(group[1:])

    return sorted(to_keep)


@ray.remote
def _process_duplicates_ray(
    size: int,
    groups: list[np.ndarray],
    coeffs: np.ndarray,
    ensure_idxs: list[int],
) -> tuple[int, list[int]]:
    """
    Process a chunk of reactions to find duplicates. This is a remote function within
    Ray.

    Args:
        size: size of reactions to process
        groups: chunks of reactions to process
        coeffs: corresponding coefficients
        ensure_idxs: indices of reactions to ensure are kept

    Returns:
        List of indices to keep
    """
    idxs_to_keep = _process_duplicates(groups, coeffs, ensure_idxs)
    return size, idxs_to_keep


def _process_duplicates(
    groups: list[np.ndarray],
    coeffs: np.ndarray,
    ensure_idxs: list[int],
) -> list[int]:
    """
    Process a chunk of reactions to find duplicates.

    Args:
        chunk: chunk of reactions to process
        coeffs: corresponding coefficients
        ensure_idxs: indices of reactions to ensure are kept
        size: size of reactions to process

    Returns:
        List of indices to keep
    """
    idxs_to_keep = []
    for group in groups:
        if group is None:
            continue
        possible_duplicate_coeffs = coeffs[group]
        relative_ensure_idxs = []
        for idx in ensure_idxs:
            if idx in group:
                relative_ensure_idxs.append(np.argwhere(group == idx).flatten()[0])
        keep_idxs = _get_idxs_to_keep(possible_duplicate_coeffs, relative_ensure_idxs)
        idxs_to_keep.extend(group[keep_idxs])

    return idxs_to_keep
