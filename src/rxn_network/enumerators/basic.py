"""
This module implements two types of basic reaction enumerators, with or without the
option of an open entry
"""
from itertools import combinations, product
from math import comb
from typing import List, Optional

from pymatgen.entries.computed_entries import ComputedEntry
from tqdm.auto import tqdm

from rxn_network.core import Calculator, Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import (
    apply_calculators,
    filter_entries_by_chemsys,
    get_elems_set,
    group_by_chemsys,
    initialize_calculators,
    initialize_entry,
    initialize_open_entries,
)
from rxn_network.reactions import ComputedReaction
from rxn_network.utils import limited_powerset


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant and product cardinality (n).
    """

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[str]] = None,
        n: int = 2,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            precursors: Optional collection of precursor formulas; only reactions
                with these phases as reactants will be enumerated.
            target: Optional formula of target; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator])
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction. Defaults to 2.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
        """
        super().__init__(precursors, target, calculators)

        self.n = n
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed

        self.stabilize = False  # whether to use only stable entries
        if "ChempotDistanceCalculator" in self.calculators:
            self.stabilize = True

    def enumerate(
        self,
        entries: GibbsEntrySet,
    ) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with specified precursors or target, the reactions will be
        filtered by these constraints.

        Args:
            entries: the set of all entries to enumerate from
        """
        entries, precursors, target = self._get_initialized_entries(entries)

        precursor_elems = get_elems_set(precursors) if precursors else None
        target_elems = {e.name for e in target.composition.elements} if target else None

        combos = list(limited_powerset(entries, self.n))
        combos_dict = group_by_chemsys(combos)
        pbar = tqdm(combos_dict.items(), desc="BasicEnumerator")

        rxns = []
        for chemsys, selected_combos in pbar:
            pbar.set_description(f"{chemsys}")
            elems = chemsys.split("-")
            if (
                (target and not target_elems.issubset(elems))
                or (precursors and not precursor_elems.issuperset(elems))
                or len(elems) >= 10
                or len(elems) == 1
            ):
                continue

            filtered_entries = filter_entries_by_chemsys(entries, chemsys)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            rxn_iter = combinations(selected_combos, 2)
            rxns.extend(self._get_rxns(rxn_iter, precursors, target, calculators))

        return list(set(rxns))

    def estimate_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return sum([comb(len(entries), i) for i in range(self.n)]) ** 2

    def _get_rxns(self, rxn_iter, precursors, target, calculators, open_phases=None):
        """
        Returns reactions from an iterable representing the combination of 2 sets
        of phases
        """
        rxns = []

        open_phases = open_phases if open_phases else []
        open_phases = set(open_phases)

        for reactants, products in rxn_iter:
            r = set(reactants)
            p = set(products)
            all_phases = r | p | open_phases

            if (
                (r & p)
                or (precursors and not precursors.issubset(all_phases))
                or (target and target not in all_phases)
            ):
                continue

            forward_rxn = ComputedReaction.balance(r, p)

            if (self.remove_unbalanced and not (forward_rxn.balanced)) or (
                self.remove_changed and forward_rxn.lowest_num_errors != 0
            ):
                continue

            backward_rxn = forward_rxn.reverse()

            if not target or target in p:
                if not precursors or (r - open_phases).issubset(precursors):
                    forward_rxn = apply_calculators(forward_rxn, calculators)
                    rxns.append(forward_rxn)
            if not target or target in r:
                if not precursors or (p - open_phases).issubset(precursors):
                    backward_rxn = apply_calculators(backward_rxn, calculators)
                    rxns.append(backward_rxn)

        return rxns

    def _get_initialized_entries(self, entries):
        precursors, target = None, None
        entries_updated = entries.copy()

        if self.precursors:
            precursors = {
                initialize_entry(f, entries, self.stabilize) for f in self.precursors
            }
            for p in precursors:
                if p not in entries:
                    old_entry = entries_updated.get_min_entry_by_formula(
                        p.composition.reduced_formula
                    )
                    entries_updated.remove(old_entry)
                    entries_updated.add(p)

        if self.target:
            target = initialize_entry(self.target, entries, self.stabilize)
            if self.stabilize:
                entries_updated.remove(entries.get_min_entry_by_formula(self.target))
                entries_updated.add(target)

        if self.stabilize:
            entries_updated = entries_updated.filter_by_stability(e_above_hull=0.0)
            self.logger.info("Filtering by stable entries!")

        return entries_updated, precursors, target


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases. Note:
    this does not return OpenComputedReaction objects (this can be calculated using
    the ReactionSet class).
    """

    def __init__(
        self,
        open_entries: List[str],
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[Calculator]] = None,
        n: int = 2,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            open_entries: List of formulas of open entries (e.g. ["O2"])
            precursors: Optional list of formulas of precursor phases; only reactions
                which have these phases as reactants will be enumerated.
            target: Optional formula of target; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator]).
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
        """
        super().__init__(
            precursors, target, calculators, n, remove_unbalanced, remove_changed
        )
        self.open_entries = open_entries

    def enumerate(self, entries: GibbsEntrySet) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with precursors/target, only reactions containing these
        phases will be enumerated. Note: this does not return
            OpenComputedReaction objects (this can be calculated using the
            ReactionSet class)

        Args:
            entries: the set of all entries to enumerate from

        Returns:
            List of unique computed reactions.
        """
        entries, precursors, target = self._get_initialized_entries(entries)

        open_entries = initialize_open_entries(self.open_entries, entries)
        open_entry_elems = get_elems_set(open_entries)

        precursor_elems = (
            get_elems_set(precursors) | open_entry_elems if precursors else None
        )
        target_elems = (
            {e.name for e in target.composition.elements} | open_entry_elems
            if target
            else None
        )

        combos = [set(c) for c in limited_powerset(entries, self.n)]

        open_combos = [
            set(c) for c in limited_powerset(open_entries, len(open_entries))
        ]
        combos_with_open = [
            combo | open_combo
            for combo in combos
            for open_combo in open_combos
            if not combo & open_combo
        ]
        combos_dict = group_by_chemsys(combos)
        combos_open_dict = group_by_chemsys(combos_with_open)
        pbar = tqdm(combos_dict.items(), desc="BasicOpenEnumerator")

        rxns = []
        for chemsys, selected_combos in pbar:
            pbar.set_description(f"{chemsys}")

            elems = chemsys.split("-")
            if (
                (chemsys not in combos_open_dict)
                or (target and not target_elems.issubset(elems))
                or (precursors and not precursor_elems.issuperset(elems))
                or len(elems) >= 10
            ):
                continue

            filtered_entries = filter_entries_by_chemsys(entries, chemsys)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            selected_open_combos = combos_open_dict[chemsys]
            rxn_iter = product(selected_combos, selected_open_combos)

            rxns.extend(
                self._get_rxns(
                    rxn_iter, precursors, target, calculators, open_phases=open_entries
                )
            )

        return list(set(rxns))
