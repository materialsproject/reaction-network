"""
This module implements two types of basic reaction enumerators, differing in the option
to consider open entries.
"""
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
from rxn_network.entries.utils import initialize_entry
from rxn_network.enumerators.utils import get_rxn_info, group_by_chemsys
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger, grouper, limited_powerset
from rxn_network.utils.ray import initialize_ray

logger = get_logger(__name__)


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n); i.e., how many phases on either side of
    the reaction. This approach does not explicitly take into account thermodynamic
    stability (i.e. phase diagram). This allows for enumeration of reactions where the
    products may not be stable with respect to each other.
    """

    CHUNK_SIZE = 2500

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_by_chemsys: Optional[str] = None,
        max_num_constraints=1,
        remove_unbalanced: bool = True,
        remove_changed: bool = True,
        calculate_e_above_hulls: bool = False,
        quiet: bool = False,
    ):
        """
        Supplied target and calculator parameters are automatically initialized as
        objects during enumeration.

        Args:
            precursors: Optional list of precursor formulas; only reactions
                which contain at least these phases as reactants will be enumerated. See
                the "exclusive_precursors" parameter for more details.
            targets: Optional list of target formulas; only reactions which include
                formation of at least one of these targets will be enumerated. See the
                "exclusive_targets" parameter for more details.
            n: Maximum reactant/product cardinality; i.e., largest possible number of
                entries on either side of the reaction. Defaults to 2.
            exclusive_precursors: Whether to consider only reactions that have reactants
                which are a subset of the provided list of precursors. In other
                words, if True, this only identifies reactions with reactants selected
                from the precursors argument. Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                form products that are a subset of the provided list of targets. If
                False, this only identifies reactions with no unspecified byproducts.
                Defualts to False.
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
        self.filter_by_chemsys = filter_by_chemsys
        self.max_num_constraints = max_num_constraints
        self.remove_unbalanced = remove_unbalanced
        self.remove_changed = remove_changed
        self.calculate_e_above_hulls = calculate_e_above_hulls
        self.quiet = quiet

        self._stabilize = False

        self._p_set_func = "issuperset" if self.exclusive_precursors else "intersection"
        self._t_set_func = "issuperset" if self.exclusive_targets else "intersection"

        self.open_phases: Optional[List] = None
        self._build_pd = False
        self._build_grand_pd = False

    def enumerate(self, entries: GibbsEntrySet, batch_size=None) -> ReactionSet:
        """
        Calculate all possible reactions given a set of entries. If the enumerator was
        initialized with specified precursors or target, the reactions will be filtered
        by these constraints. Every enumerator follows a standard procedure:

        1. Initialize entries, i.e., ensure that precursors and target are considered
        stable entries within the entry set. If using ChempotDistanceCalculator, ensure
        that entries are filtered by stability.

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

        initialize_ray()

        entries, precursors, targets, open_entries = self._get_initialized_entries(
            entries
        )

        combos_dict = self._get_combos_dict(
            entries,
            precursors,
            targets,
            open_entries,
        )
        open_combos = self._get_open_combos(open_entries)

        if not open_combos:
            open_combos = []

        items = combos_dict.items()

        precursors = ray.put(precursors)
        targets = ray.put(targets)
        react_function = ray.put(self._react_function)
        open_entries = ray.put(open_entries)
        p_set_func = ray.put(self._p_set_func)
        t_set_func = ray.put(self._t_set_func)
        remove_unbalanced = ray.put(self.remove_unbalanced)
        remove_changed = ray.put(self.remove_changed)
        max_num_constraints = ray.put(self.max_num_constraints)

        rxn_chunk_refs = []  # type: ignore
        results = []

        if not batch_size:
            batch_size = ray.cluster_resources()["CPU"] * 2

        with tqdm(
            total=self._num_chunks(items, open_combos), disable=self.quiet
        ) as pbar:
            for item in items:
                chemsys, combos = item

                elems = chemsys.split("-")

                filtered_entries = None
                pd = None
                grand_pd = None

                if self.build_pd or self.build_grand_pd:
                    filtered_entries = entries.get_subset_in_chemsys(elems)

                if self.build_pd:
                    pd = PhaseDiagram(filtered_entries)

                if self.build_grand_pd:
                    chempots = getattr(self, "chempots")
                    grand_pd = GrandPotentialPhaseDiagram(filtered_entries, chempots)

                filtered_entries = ray.put(filtered_entries)
                pd = ray.put(pd)
                grand_pd = ray.put(grand_pd)

                for rxn_iterable_chunk in grouper(
                    self._get_rxn_iterable(combos, open_combos), self.CHUNK_SIZE
                ):
                    if len(rxn_chunk_refs) > batch_size:
                        num_ready = len(rxn_chunk_refs) - batch_size
                        newly_completed, rxn_chunk_refs = ray.wait(
                            rxn_chunk_refs, num_returns=num_ready
                        )
                        for completed_ref in newly_completed:
                            results.extend(ray.get(completed_ref))

                            pbar.update(1)

                    rxn_chunk_refs.append(
                        _react.remote(
                            rxn_iterable_chunk,
                            react_function,
                            open_entries,
                            precursors,
                            targets,
                            p_set_func,
                            t_set_func,
                            remove_unbalanced,
                            remove_changed,
                            max_num_constraints,
                            filtered_entries,
                            pd,
                            grand_pd,
                        )
                    )

            newly_completed, rxn_chunk_refs = ray.wait(
                rxn_chunk_refs, num_returns=len(rxn_chunk_refs)
            )
            for completed_ref in newly_completed:
                results.extend(ray.get(completed_ref))
                pbar.update(1)

        all_indices, all_coeffs, all_data = [], [], []
        for r in results:
            all_indices.append(r[0])
            all_coeffs.append(r[1])
            all_data.append(r[2])

        rxn_set = ReactionSet(
            entries.entries_list, all_indices, all_coeffs, all_data=all_data
        )
        rxn_set = rxn_set.filter_duplicates()

        return rxn_set

    @classmethod
    def _num_chunks(cls, items, open_combos):
        _ = open_combos  # not used

        n = 0
        for _, i in items:
            num_combos = cls._rxn_iter_length(i, open_combos)
            if num_combos > 0:
                n += num_combos // cls.CHUNK_SIZE + 1

        return n

    @staticmethod
    def _rxn_iter_length(combos, open_combos):
        _ = open_combos  # not used
        return comb(len(combos), 2)

    def _get_combos_dict(
        self, entries, precursor_entries, target_entries, open_entries
    ):
        """
        Gets all possible entry combinations up to predefined cardinality (n), filtered
        and grouped by chemical system.
        """
        precursor_elems = [
            [str(el) for el in e.composition.elements] for e in precursor_entries
        ]
        target_elems = [
            [str(el) for el in e.composition.elements] for e in target_entries
        ]
        all_open_elems = {el for e in open_entries for el in e.composition.elements}

        entries = entries - open_entries

        combos = [set(c) for c in limited_powerset(entries, self.n)]
        combos_dict = group_by_chemsys(combos, all_open_elems)

        filtered_combos = self._filter_dict_by_elems(
            combos_dict,
            precursor_elems,
            target_elems,
            all_open_elems,
        )

        return filtered_combos

    def _get_open_combos(  # pylint: disable=useless-return
        self, open_entries
    ) -> Optional[List[Set[ComputedEntry]]]:
        """No open entries for BasicEnumerator, returns None"""
        _ = (self, open_entries)  # unused_arguments
        return None

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
            logger.info("Filtering by stable entries!")

        entries_new.build_indices()

        open_entries = set()
        if self.open_phases:
            open_entries = {
                e
                for e in entries_new
                if e.composition.reduced_formula in self.open_phases
            }

        return entries_new, precursors, targets, open_entries

    def _filter_dict_by_elems(
        self,
        combos_dict,
        precursor_elems,
        target_elems,
        all_open_elems,
    ):
        """Filters the dictionary of combinations by elements"""
        filtered_dict = {}

        all_precursor_elems = {el for g in precursor_elems for el in g}
        all_target_elems = {el for g in target_elems for el in g}
        all_open_elems = {str(el) for el in all_open_elems}

        filter_elems = None
        if self.filter_by_chemsys:
            filter_elems = set(self.filter_by_chemsys.split("-"))

        for chemsys, combos in combos_dict.items():
            elems = set(chemsys.split("-"))

            if filter_elems:
                if not elems.issuperset(filter_elems):
                    continue

            if len(elems) >= 10 or len(elems) == 1:  # too few or too many elements
                continue

            if precursor_elems:
                if not getattr(all_precursor_elems | all_open_elems, self._p_set_func)(
                    elems
                ):
                    continue

            if target_elems:
                if not getattr(all_target_elems | all_open_elems, self._t_set_func)(
                    elems
                ):
                    continue

            filtered_dict[chemsys] = combos

        return filtered_dict

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

    CHUNK_SIZE = 2500

    def __init__(
        self,
        open_phases: List[str],
        precursors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        n: int = 2,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        filter_by_chemsys: Optional[str] = None,
        max_num_constraints: int = 1,
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
            targets: Optional list of formulas of targets; only reactions
                which make this target will be enumerated.
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
            filter_by_chemsys=filter_by_chemsys,
            max_num_constraints=max_num_constraints,
            remove_unbalanced=remove_unbalanced,
            remove_changed=remove_changed,
        )
        self.open_phases: List[str] = open_phases

    @staticmethod
    def _rxn_iter_length(combos, open_combos):
        num_combos_with_open = sum(
            1 if not i & j else 0 for i in combos for j in open_combos
        )

        return len(combos) * num_combos_with_open

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


@ray.remote
def _react(
    rxn_iterable,
    react_function,
    open_entries,
    precursors,
    targets,
    p_set_func,
    t_set_func,
    remove_unbalanced,
    remove_changed,
    max_num_constraints,
    filtered_entries,
    pd,
    grand_pd,
):
    """
    This function is a wrapper for the specific react function of each enumerator. This
    wrapper contains the logic used for filtering out reactions based on the
    user-defined enumerator settings. It can also be called as a remote function using
    ray, allowing for parallel computation during reaction enumeration.

    Note: this function is not intended to to be called directly!

    """
    all_rxns = []

    for rp in rxn_iterable:
        if not rp:
            continue

        r = set(rp[0]) if rp[0] else set()
        p = set(rp[1]) if rp[1] else set()

        all_phases = r | p

        precursor_func = (
            getattr(precursors | open_entries, p_set_func)
            if precursors
            else lambda e: True
        )
        target_func = (
            getattr(targets | open_entries, t_set_func) if targets else lambda e: True
        )

        if (
            (r & p)
            or (precursors and not precursors & all_phases)
            or (p and targets and not targets & all_phases)
        ):
            continue

        if not (precursor_func(r) or (p and precursor_func(p))):
            continue
        if p and not (target_func(r) or target_func(p)):
            continue

        suggested_rxns = react_function(
            r, p, filtered_entries=filtered_entries, pd=pd, grand_pd=grand_pd
        )

        rxns = []
        for rxn in suggested_rxns:
            if (
                rxn.is_identity
                or (remove_unbalanced and not rxn.balanced)
                or (remove_changed and rxn.lowest_num_errors != 0)
                or rxn.data["num_constraints"] > max_num_constraints
            ):
                continue

            reactant_entries = set(rxn.reactant_entries) - open_entries
            product_entries = set(rxn.product_entries) - open_entries

            if precursor_func(reactant_entries) and target_func(product_entries):
                rxns.append(get_rxn_info(rxn))

        all_rxns.extend(rxns)

    return all_rxns
