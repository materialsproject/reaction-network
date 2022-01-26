"""
This module implements two types of reaction enumerators using a free energy
minimization technique, with or without the option of an open entry.
"""

from itertools import product
from math import comb
from typing import List, Optional

from pymatgen.analysis.interface_reactions import (
    GrandPotentialInterfacialReactivity,
    InterfacialReactivity,
)
from pymatgen.core.composition import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

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
        targets: Optional[List[str]] = None,
        calculators: Optional[List[str]] = None,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        quiet: bool = False,
    ):
        """
        Args:
            precursors: Optional formulas of precursors.
            targets: Optional formulas of targets; only reactions which make these targets
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator"])
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """
        super().__init__(
            precursors=precursors,
            targets=targets,
            calculators=calculators,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            quiet=quiet,
        )
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
        """React method for MinimizeGibbsEnumerator, which uses the interfacial reaction
        approach (see _react_interface())"""
        r = list(reactants)
        r0 = r[0]

        if len(r) == 1:
            r1 = r[0]
        else:
            r1 = r[1]

        return self._react_interface(
            r0.composition, r1.composition, pd, grand_pd, calculators=calculators
        )

    @staticmethod
    def _get_rxn_iterable(combos, open_combos):
        """Gets the iterable used to generate reactions"""
        _ = open_combos  # unused argument

        return product(combos, [None])

    @staticmethod
    def _react_interface(r1, r2, pd, grand_pd=None, calculators=None):
        """Simple API for InterfacialReactivity module from pymatgen."""
        chempots = None

        if grand_pd:
            interface = GrandPotentialInterfacialReactivity(
                r1,
                r2,
                grand_pd,
                pd_non_grand=pd,
                norm=True,
                include_no_mixing_energy=True,
                use_hull_energy=True,
            )
            chempots = grand_pd.chempots

        else:
            interface = InterfacialReactivity(
                r1,
                r2,
                pd,
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
        targets: Optional[List[str]] = None,
        calculators: Optional[List[str]] = None,
        exclusive_precursors: bool = True,
        exclusive_targets: bool = False,
        quiet: bool = False,
    ):
        """
        Args:
            open_elem: The element to be considered as open
            mu: The chemical potential of the open element
            precursors: Optional formulas of precursors.
            targets: Optional formulas of targets; only reactions which make these targets
                will be enumerated.
            calculators: Optional list of Calculator object names; see calculators
                module for options (e.g., ["ChempotDistanceCalculator])
            exclusive_precursors: Whether to consider only reactions that have
                reactants which are a subset of the provided list of precursors.
                Defaults to True.
            exclusive_targets: Whether to consider only reactions that make the
                provided target directly (i.e. with no byproducts). Defualts to False.
            quiet: Whether to run in quiet mode (no progress bar). Defaults to False.
        """

        super().__init__(
            precursors=precursors,
            targets=targets,
            calculators=calculators,
            exclusive_precursors=exclusive_precursors,
            exclusive_targets=exclusive_targets,
            quiet=quiet,
        )
        self.open_elem = Element(open_elem)  # type: ignore
        self.open_phases = [Composition(str(self.open_elem)).reduced_formula]  # type: ignore
        self.mu = mu
        self.chempots = {self.open_elem: self.mu}
        self._build_grand_pd = True

    def _react(self, reactants, products, calculators, pd=None, grand_pd=None):
        """Same as the MinimizeGibbsEnumerator react function, but with ability to
        specify open element and grand potential phase diagram"""
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
