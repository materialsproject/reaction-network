"""
This module implements two types of basic reaction enumerators, with or without the
option of an open entry
"""
from itertools import combinations, product
from math import comb
from typing import List, Optional

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from tqdm.auto import tqdm

from rxn_network.core import Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import (
    apply_calculators,
    filter_entries_by_chemsys,
    get_elems_set,
    group_by_chemsys,
    initialize_calculators,
    initialize_entry,
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
            precursors: Optional list of precursor formulas; only reactions
                which contain at least these phases as reactants will be enumerated.
            target: Optional formula of target; only reactions which include
                formation of this target will be enumerated.
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

        self._stabilize = False
        if "ChempotDistanceCalculator" in self.calculators:
            self._stabilize = True

        self._build_pd = False
        self._build_grand_pd = False

    def enumerate(self, entries: GibbsEntrySet) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with specified precursors or target, the reactions will be
        filtered by these constraints. Every enumerator follows a standard procedure:

        1. Initialize entries, i.e. ensure that precursors and target are considered
        stable entries within the entry set. If using ChempotDistanceCalculator,
        ensure that entries are filtered by stability.

        2. Get a dictionary representing every possible node, i.e. phase combination,
        grouped by chemical system.

        3. Filter the combos dictionary for chemical systems which are not relevant (
        i.e. don't contain elements in precursors and/or target.

        4. Iterate through each chemical system, initializing calculators,
        and computing all possible reactions for reactant/product pair and/or
        thermodynamically predicted reactions for given reactants.

        5. Add reactions to growing list, repeat Step 4 until combos dict exhausted.

        Args:
            entries: the set of all entries to enumerate from
        """
        entries, precursors, target = self._get_initialized_entries(entries)
        combos_dict = self._get_combos_dict(entries, precursors, target)
        open_combos = self._get_open_combos(entries)

        pbar = tqdm(combos_dict.items(), desc=self.__class__.__name__)

        rxns = []
        for chemsys, combos in pbar:
            pbar.set_description(f"{chemsys}")

            filtered_entries = filter_entries_by_chemsys(entries, chemsys)
            calculators = initialize_calculators(self.calculators, filtered_entries)

            rxn_iter = self._get_rxn_iterable(combos, open_combos)
            r = self._get_rxns(
                rxn_iter, precursors, target, calculators, filtered_entries
            )
            rxns.extend(r)

        return list(set(rxns))

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return sum([comb(len(entries), i) for i in range(1, self.n + 1)]) ** 2

    def _get_combos_dict(self, entries, precursor_entries, target_entries):
        precursor_elems = (
            get_elems_set(precursor_entries) if precursor_entries else set()
        )
        target_elems = (
            {e.name for e in target_entries.composition.elements}
            if target_entries
            else set()
        )

        combos = [set(c) for c in limited_powerset(entries, self.n)]

        combos_dict = group_by_chemsys(combos)
        filtered_combos = self._filter_dict_by_elems(
            combos_dict, precursor_elems, target_elems
        )

        return filtered_combos

    def _get_open_combos(self, entries):
        """ No open entries for BasicEnumerator, returns None"""
        return None

    def _get_rxns(
        self, rxn_iterable, precursors, target, calculators, filtered_entries=None
    ):
        """
        Returns reactions from an iterable representing the combination of 2 sets
        of phases. Works for reactions with open phases.
        """
        pd = None
        if self.build_pd:
            pd = PhaseDiagram(filtered_entries)

        grand_pd = None
        if self.build_grand_pd:
            chempots = getattr(self, "chempots")
            grand_pd = GrandPotentialPhaseDiagram(filtered_entries, chempots)

        rxns = []
        for reactants, products in rxn_iterable:
            r = set(reactants) if reactants else set()
            p = set(products) if products else set()
            all_phases = r | p

            if (r & p) or (precursors and not precursors.issubset(all_phases)):
                continue

            suggested_rxns = self._react(r, p, calculators, pd, grand_pd)

            for rxn in suggested_rxns:
                if (
                    rxn.is_identity
                    or (self.remove_unbalanced and not rxn.balanced)
                    or (self.remove_changed and rxn.lowest_num_errors != 0)
                ):
                    continue

                if not target or target in rxn.product_entries:
                    if not precursors or precursors.issubset(rxn.reactant_entries):
                        rxns.append(rxn)

        return rxns

    def _react(self, reactants, products, calculators, pd=None, grand_pd=None):
        forward_rxn = ComputedReaction.balance(reactants, products)
        backward_rxn = forward_rxn.reverse()

        forward_rxn = apply_calculators(forward_rxn, calculators)
        backward_rxn = apply_calculators(backward_rxn, calculators)
        return [forward_rxn, backward_rxn]

    def _get_rxn_iterable(self, combos, open_combos):
        return combinations(combos, 2)

    def _get_initialized_entries(self, entries):
        precursors, target = None, None
        entries_new = entries.copy()

        if self.precursors:
            precursors = {
                initialize_entry(f, entries, self.stabilize) for f in self.precursors
            }
            for p in precursors:
                if p not in entries:
                    old_entry = entries_new.get_min_entry_by_formula(
                        p.composition.reduced_formula
                    )
                    entries_new.remove(old_entry)
                    entries_new.add(p)

        if self.target:
            target = initialize_entry(self.target, entries, self.stabilize)
            if self.stabilize:
                entries_new.remove(entries.get_min_entry_by_formula(self.target))
                entries_new.add(target)

        if self.stabilize:
            entries_new = entries_new.filter_by_stability(e_above_hull=0.0)
            self.logger.info("Filtering by stable entries!")

        return entries_new, precursors, target

    @staticmethod
    def _filter_dict_by_elems(combos_dict, precursor_elems, target_elems):
        filtered_dict = dict()
        for chemsys, combos in combos_dict.items():
            elems = chemsys.split("-")
            if (
                (target_elems and not target_elems.issubset(elems))
                or (precursor_elems and not precursor_elems.issubset(elems))
                or len(elems) >= 10
                or len(elems) == 1
            ):
                continue
            else:
                filtered_dict[chemsys] = combos

        return filtered_dict

    @property
    def stabilize(self):
        """ Whether or not to use only stable entries in analysis"""
        return self._stabilize

    @property
    def build_pd(self):
        """Whether or not to build a PhaseDiagram object during reaction enumeration (
        useful for some analyses)"""
        return self._build_pd

    @property
    def build_grand_pd(self):
        """Whether or not to build a GrandPotentialPhaseDiagram object during
        reaction enumeration (useful for some analyses)"""
        return self._build_grand_pd


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open phases. Note:
    this does not return OpenComputedReaction objects (this can be calculated using
    the ReactionSet class).
    """

    def __init__(
        self,
        open_phases: List[str],
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
            open_phases: List of formulas of open entries (e.g. ["O2"])
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
        self.open_phases = open_phases

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        num_open_phases = len(self.open_phases)
        num_combos_with_open = sum(
            [comb(num_open_phases, i) for i in range(1, num_open_phases + 1)]
        )

        num_total_combos = 0
        for i in range(1, self.n + 1):
            num_combos = comb(len(entries), i)
            num_total_combos += num_combos_with_open * num_combos

        return num_total_combos ** 2

    def _get_open_combos(self, entries):
        open_entries = {
            e for e in entries if e.composition.reduced_formula in self.open_phases
        }
        open_combos = [
            set(c) for c in limited_powerset(open_entries, len(open_entries))
        ]
        return open_combos

    def _get_rxn_iterable(self, combos, open_combos):
        combos_with_open = [
            combo | open_combo
            for combo in combos
            for open_combo in open_combos
            if not combo & open_combo
        ]
        rxn_iter = product(combos, combos_with_open)

        return rxn_iter
