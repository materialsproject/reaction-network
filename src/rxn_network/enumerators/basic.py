"""
This module implements two types of basic reaction enumerators, differing in the option
to consider open entries.
"""
import logging
from copy import deepcopy
from itertools import combinations, product
from math import comb
from typing import List, Optional, Set

import ray
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from tqdm import tqdm

from rxn_network.core.enumerator import Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import group_by_chemsys, initialize_entry, react
from rxn_network.reactions import ComputedReaction
from rxn_network.utils import grouper, initialize_ray, limited_powerset, to_iterator


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n); i.e., how many phases on either side of
    the reaction. This approach does not explicitly take into account thermodynamic
    stability (i.e. phase diagram). This allows for enumeration of reactions where the
    products may not be stable with respect to each other.
    """

    CHUNK_SIZE = 5000

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        calculate_e_above_hulls: bool = True,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            precursors: Optional list of precursor formulas; only reactions
                which contain at least these phases as reactants will be enumerated. See the
                "exclusive_precursors" parameter for more details.
            targets: Optional list of target formulas; only reactions which include
                formation of at least one of these targets will be enumerated. See the
                "exclusive_targets" parameter for more details.
            calculators: Optional list of Calculator object names to be initialized; see calculators
                module for options (e.g., ["ChempotDistanceCalculator"])
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction. Defaults to 2.
            exclusive_precursors: Whether to consider only reactions that have
            reactants which are a subset of the provided list of precursors. In other
                words, if True, this only identifies reactions with reactants selected from the precursors
                argument. Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                form products that are a subset of the provided list of targets. If False,
                this only identifies reactions with no unspecified byproducts. Defualts to False.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True.
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
            calculate_e_above_hulls: Whether to calculate e_above_hull for each entry
                upon initialization of the entries at the beginning of enumeration.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(precursors=precursors, targets=targets)

        self.n = n
        self.exclusive_precursors = exclusive_precursors
        self.exclusive_targets = exclusive_targets
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed
        self.calculate_e_above_hulls = calculate_e_above_hulls
        self.quiet = quiet

        self._stabilize = False

        self.open_phases: Optional[List] = None
        self._build_pd = False
        self._build_grand_pd = False
        self.logger = logging.Logger("enumerator")

    def enumerate(self, entries: GibbsEntrySet) -> List[ComputedReaction]:
        """
        Calculate all possible reactions given a set of entries. If the enumerator
        was initialized with specified precursors or target, the reactions will be
        filtered by these constraints. Every enumerator follows a standard procedure:

        1. Initialize entries, i.e., ensure that precursors and target are considered
        stable entries within the entry set. If using ChempotDistanceCalculator,
        ensure that entries are filtered by stability.

        2. Get a dictionary representing every possible "node", i.e. phase combination,
        grouped by chemical system.

        3. Filter the combos dictionary for chemical systems which are not relevant;
        i.e., don't contain elements in precursors and/or target.

        4. Iterate through each chemical system, initializing calculators,
        and computing all possible reactions for reactant/product pair and/or
        thermodynamically predicted reactions for given reactants.

        5. Add reactions to growing list, repeat Step 4 until combos dict exhausted.

        Args:
            entries: the set of all entries to enumerate from
        """

        initialize_ray(quiet=self.quiet)

        entries, precursors, targets, open_entries = self._get_initialized_entries(
            entries
        )

        combos_dict = self._get_combos_dict(entries, precursors, targets, open_entries)
        open_combos = self._get_open_combos(open_entries)
        if not open_combos:
            open_combos = []

        items = combos_dict.items()

        precursors = ray.put(precursors)
        targets = ray.put(targets)
        react_function = ray.put(self._react_function)

        rxns = []

        for item in items:
            rxns.extend(
                self._get_rxns_in_chemsys(
                    item,
                    open_combos,
                    react_function,
                    entries,
                    open_entries,
                    precursors,
                    targets,
                )
            )

        results = []

        iterator = to_iterator(rxns)
        if not self.quiet:
            iterator = tqdm(iterator, total=len(rxns), disable=self.quiet)

        for r in iterator:
            results.extend(r)

        return list(set(results))

    def _get_combos_dict(
        self, entries, precursor_entries, target_entries, open_entries
    ):
        """
        Gets all possible entry combinations up to predefined cardinality n, filtered and
        grouped by chemical system
        """
        precursor_elems = [
            [str(el) for el in e.composition.elements] for e in precursor_entries
        ]
        target_elems = [
            [str(el) for el in e.composition.elements] for e in target_entries
        ]
        open_elems = [[str(el) for el in e.composition.elements] for e in open_entries]

        entries = entries - open_entries

        combos = []
        for c in limited_powerset(entries, self.n):
            c_set = set(c)
            combos.append(c_set)

        combos_dict = group_by_chemsys(combos, [el for g in open_elems for el in g])

        filtered_combos = self._filter_dict_by_elems(
            combos_dict, precursor_elems, target_elems, open_elems
        )

        return filtered_combos

    def _get_open_combos(  # pylint: disable=R1711
        self, open_entries
    ) -> Optional[List[Set[ComputedEntry]]]:
        """No open entries for BasicEnumerator, returns None"""
        _ = (self, open_entries)  # unused_arguments
        return None

    def _get_rxns_in_chemsys(
        self,
        item,
        open_combos,
        react_function,
        entries,
        open_entries,
        precursors,
        targets,
    ):
        """ """
        chemsys, combos = item

        elems = chemsys.split("-")
        filtered_entries = entries.get_subset_in_chemsys(elems)

        rxn_iter = self._get_rxn_iterable(combos, open_combos)

        rxns = self._get_rxns_from_iterable(
            rxn_iter,
            precursors,
            targets,
            react_function,
            filtered_entries,
            open_entries,
        )
        return rxns

    def _get_rxns_from_iterable(
        self,
        rxn_iterable,
        precursors,
        targets,
        react_function,
        filtered_entries=None,
        open_entries=None,
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

        p_set_func = "issuperset" if self.exclusive_precursors else "intersection"
        t_set_func = "issuperset" if self.exclusive_targets else "intersection"

        if not open_entries:
            open_entries = set()

        open_entries = ray.put(open_entries)
        p_set_func = ray.put(p_set_func)
        t_set_func = ray.put(t_set_func)
        remove_unbalanced = ray.put(self.remove_unbalanced)
        remove_changed = ray.put(self.remove_changed)
        filtered_entries = ray.put(filtered_entries)
        pd = ray.put(pd)
        grand_pd = ray.put(grand_pd)

        rxns = [
            react.remote(
                rxn_iterable_chunk,
                react_function,
                open_entries,
                precursors,
                targets,
                p_set_func,
                t_set_func,
                remove_unbalanced,
                remove_changed,
                filtered_entries,
                pd,
                grand_pd,
            )
            for rxn_iterable_chunk in grouper(rxn_iterable, self.CHUNK_SIZE)
        ]
        return rxns

    @staticmethod
    def _react_function(reactants, products, **kwargs):
        _ = kwargs  # unused_argument
        forward_rxn = ComputedReaction.balance(reactants, products)
        backward_rxn = forward_rxn.reverse()
        return [forward_rxn, backward_rxn]

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Get all reaction/product combinations"""
        _ = open_combos  # unused argument

        return combinations(combos, 2)

    def _get_initialized_entries(self, entries):
        """Returns initialized entries, precursors, target, and open entries"""

        def initialize_entries_list(ents):
            new_ents = {initialize_entry(f, entries, self.stabilize) for f in ents}
            return new_ents

        precursors, targets = set(), set()

        entries_new = GibbsEntrySet(
            deepcopy(entries), calculate_e_above_hulls=self.calculate_e_above_hulls
        )

        if self.precursors:
            precursors = initialize_entries_list(self.precursors)
        if self.targets:
            targets = initialize_entries_list(self.targets)

        for e in precursors | targets:
            if e not in entries_new:
                try:
                    old_e = entries_new.get_min_entry_by_formula(
                        e.composition.reduced_formula
                    )
                    entries_new.discard(old_e)
                except KeyError:
                    pass

                entries_new.add(e)

        if self.stabilize:
            entries_new = entries_new.filter_by_stability(e_above_hull=0.0)
            self.logger.info("Filtering by stable entries!")

        open_entries = set()
        if self.open_phases:
            open_entries = {
                e for e in entries if e.composition.reduced_formula in self.open_phases
            }

        return entries_new, precursors, targets, open_entries

    def _filter_dict_by_elems(
        self,
        combos_dict,
        precursor_elems,
        target_elems,
        open_elems,
    ):
        """Filters the dictionary of combinations by elements"""
        filtered_dict = {}

        all_precursor_elems = {el for g in precursor_elems for el in g}
        all_target_elems = {el for g in target_elems for el in g}

        for chemsys, combos in combos_dict.items():
            elems = set(chemsys.split("-"))

            if len(elems) >= 10 or len(elems) == 1:
                continue

            all_open_elems = {el for g in open_elems for el in g}
            if not all_open_elems.issubset(elems):
                continue

            if precursor_elems:
                if self.exclusive_precursors:
                    if not all_precursor_elems == elems:
                        continue
                else:
                    if not any(
                        elems.issuperset(el_group) for el_group in precursor_elems
                    ):
                        continue

            if target_elems:
                if self.exclusive_targets:
                    if not all_target_elems == elems:
                        continue
                else:
                    if not any(elems.issuperset(el_group) for el_group in target_elems):
                        continue

            filtered_dict[chemsys] = combos

        return filtered_dict

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return sum([comb(len(entries), i) for i in range(1, self.n + 1)]) ** 2

    @property
    def stabilize(self):
        """Whether or not to use only stable entries in analysis"""
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

    CHUNK_SIZE = 1000

    def __init__(
        self,
        open_phases: List[str],
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        calculate_e_above_hulls: bool = False,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            open_phases: List of formulas of open entries (e.g. ["O2"])
            precursors: Optional list of formulas of precursor phases; only reactions
                which have these phases as reactants will be enumerated.
            targets: Optional list of formulas of targets; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator]).
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction.
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            remove_unbalanced: Whether to remove reactions which are unbalanced.
                Defaults to True
            remove_changed: Whether to remove reactions which can only be balanced by
                removing a reactant/product or having it change sides. Defaults to True.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
            n=n,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
        )
        self.open_phases: List[str] = open_phases

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

        return num_total_combos**2

    def _get_open_combos(self, open_entries):
        """Get all possible combinations of open entries. For a single entry,
        this is just the entry itself."""
        open_combos = [
            set(c) for c in limited_powerset(open_entries, len(open_entries))
        ]
        return open_combos

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Get all reaction/product combinations."""
        combos_with_open = [
            combo | open_combo
            for combo in combos
            for open_combo in open_combos
            if not combo & open_combo
        ]
        rxn_iter = product(combos, combos_with_open)

        return rxn_iter
