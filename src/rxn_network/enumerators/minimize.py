from itertools import combinations, product
import math
from typing import List, Optional

from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry
from tqdm.auto import tqdm

from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.utils import (
    apply_calculators,
    filter_entries_by_chemsys,
    get_computed_rxn,
    get_elems_set,
    get_open_computed_rxn,
    group_by_chemsys,
    initialize_calculators,
    initialize_entry,
)
from rxn_network.reactions import ComputedReaction


class MinimizeGibbsEnumerator(BasicEnumerator):
    """
    Enumerator for finding all reactions between two reactants that are predicted by
    thermodynamics; i.e., they appear when taking the convex hull along a straight
    line connecting any two phases in G-x phase space. Identity reactions are
    automatically excluded.
    """

    def __init__(
        self,
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[str]] = None,
    ):
        """
        Args:
            precursors: Optional formulas of precursors.
            target: Optional formula of target; only reactions which make this target
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator])
        """
        super().__init__(precursors, target, calculators)

    def estimate_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return math.comb(len(entries), 2)

    def _react(self, reactants, products, calculators, pd=None):
        e1 = reactants[0]
        e2 = reactants[1]

        return self._react_interface(
            e1.composition, e2.composition, pd, calculators=calculators
        )

    def _get_rxn_iter(self, combos):
        return product(combos, None)

    def _react_interface(
        self, r1, r2, pd, grand_pd=None, open_entry=None, calculators=None
    ):
        "Simple API for InterfacialReactivity module from pymatgen."
        if grand_pd:
            interface = InterfacialReactivity(
                r1,
                r2,
                grand_pd,
                norm=True,
                include_no_mixing_energy=False,
                pd_non_grand=pd,
                use_hull_energy=True,
            )
            chempots = grand_pd.chempots

        else:
            interface = InterfacialReactivity(
                r1,
                r2,
                pd,
                norm=False,
                include_no_mixing_energy=False,
                pd_non_grand=None,
                use_hull_energy=True,
            )

        rxns = []
        for _, _, _, rxn, _ in interface.get_kinks():
            if grand_pd:
                rxn = get_open_computed_rxn(rxn, pd.all_entries, chempots)
            else:
                rxn = get_computed_rxn(rxn, pd.all_entries)

            if rxn.is_identity or rxn.lowest_num_errors > 0:
                continue

            if self.target and self.target not in rxn.product_entries:
                continue

            rxn = apply_calculators(rxn, calculators)
            rxns.append(rxn)

        return rxns


class MinimizeGrandPotentialEnumerator(MinimizeGibbsEnumerator):
    """
    Enumerator for finding all reactions between two reactants and an open element
    that are predicted by thermo; i.e., they appear when taking the
    convex hull along a straight line connecting any two phases in Phi-x
    phase space. Identity reactions are excluded.
    """

    def __init__(
        self,
        open_elem: Element,
        mu: float,
        precursors: Optional[List[str]] = None,
        target: Optional[str] = None,
        calculators: Optional[List[str]] = None,
    ):
        super().__init__(precursors=precursors, target=target, calculators=calculators)
        self.open_elem = Element(open_elem)  # type: ignore
        self.mu = mu

    def enumerate(self, entries) -> List[ComputedReaction]:
        entries = GibbsEntrySet(entries)

        target = None
        if self.target:
            target = initialize_entry(self.target, entries)
            entries.add(target)
            target_elems = {str(e) for e in target.composition.elements}

        precursors = None
        if self.precursors:
            precursors = {initialize_entry(f, entries) for f in self.precursors}
            for p in precursors:
                entries.add(p)
            precursor_elems = {
                str(elem) for p in precursors for elem in p.composition.elements
            }

        if "ChempotDistanceCalculator" in self.calculators:
            entries = entries.filter_by_stability(e_above_hull=0.0)
            self.logger.info(
                "Filtering by stable entries due to use of 'ChempotDistanceCalculator'"
            )

        open_entry = sorted(
            filter(lambda e: e.composition.elements == [self.open_elem], entries),
            key=lambda e: e.energy_per_atom,
        )[0]
        entries_no_open = GibbsEntrySet(entries)
        entries_no_open.remove(open_entry)
        combos = list(combinations(entries_no_open, 2))
        combos_dict = group_by_chemsys(combos, self.open_elem)

        rxns = []
        for chemsys, combos in tqdm(combos_dict.items()):
            elems = chemsys.split("-")
            if (
                (target and not target_elems.issubset(elems))
                or (precursors and not precursor_elems.issuperset(elems))
                or len(elems) >= 10
            ):
                continue

            chemsys_entries = filter_entries_by_chemsys(entries, chemsys)
            pd = PhaseDiagram(chemsys_entries)

            calculators = initialize_calculators(self.calculators, chemsys_entries)

            grand_pd = GrandPotentialPhaseDiagram(
                chemsys_entries, {self.open_elem: self.mu}
            )
            for e1, e2 in combos:
                if precursors and {e1, e2} != precursors:
                    continue
                predicted_rxns = self._react_interface(
                    e1.composition,
                    e2.composition,
                    pd,
                    grand_pd,
                    open_entry,
                    calculators,
                )
                rxns.extend(predicted_rxns)

        return list(set(rxns))
