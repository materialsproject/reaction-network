from itertools import combinations, product
from math import comb
from typing import List, Optional

from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry
from tqdm.auto import tqdm

from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.utils import (
    apply_calculators,
    get_computed_rxn,
    get_open_computed_rxn,
)


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
        self._build_pd = True

    def estimate_max_num_reactions(self, entries: List[ComputedEntry]) -> int:
        """
        Estimate the upper bound of the number of possible reactions. This will
        correlate with the amount of time it takes to enumerate reactions.

        Args:
            entries: A list of all entries to consider

        Returns: The upper bound on the number of possible reactions
        """
        return comb(len(entries), 2) * 2

    def _react(self, reactants, products, calculators, pd=None, grand_pd=None):
        r = list(reactants)
        r0 = r[0]

        if len(r) == 1:
            r1 = r[0]
        else:
            r1 = r[1]

        return self._react_interface(
            r0.composition, r1.composition, pd, grand_pd, calculators=calculators
        )

    def _get_rxn_iterable(self, combos, open_entries):
        return product(combos, [None])

    def _react_interface(self, r1, r2, pd, grand_pd=None, calculators=None):
        """Simple API for InterfacialReactivity module from pymatgen."""
        chempots = None

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
        self.chempots = {self.open_elem: self.mu}
        self._build_grand_pd = True

    def _react(self, reactants, products, calculators, pd=None, grand_pd=None):
        r = list(reactants)
        r0 = r[0]

        if len(r) == 1:
            r1 = r[0]
        else:
            r1 = r[1]

        open_elem = list(grand_pd.chempots.keys())[0]

        for reactant in r:
            elems = reactant.composition.elements
            if len(elems) == 1 and elems[0] == open_elem:  # skip if reactant = open_e
                return []

        return self._react_interface(
            r0.composition,
            r1.composition,
            pd,
            grand_pd=grand_pd,
            calculators=calculators,
        )
