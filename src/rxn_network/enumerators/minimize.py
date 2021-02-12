from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from pymatgen.analysis.interface_reactions import InterfacialReactivity

from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import (
    get_total_chemsys,
    group_by_chemsys,
    filter_entries_by_chemsys,
    get_entry_by_comp,
    get_computed_rxn
)


class MinimizeGibbsEnumerator(Enumerator):
    """
    Enumerator for finding all reactions between two reactants (+ optional open
    element) that are predicted by thermodynamics, i.e., they appear when taking the
    convex hull along a straight line connecting any two phases in G-x
    phase space.
    """

    def __init__(self):
        pass

    def enumerate(self, entries):
        combos = list(combinations(entries, 2))
        combos_dict = group_by_chemsys(combos)

        rxns = []
        for chemsys, combos in combos_dict.items():
            chemsys_entries = filter_entries_by_chemsys(entries, chemsys)
            pd = PhaseDiagram(chemsys_entries)
            for e1, e2 in combos:
                predicted_rxns = self._react_interface(e1.composition, e2.composition,
                                                      pd)
                rxns.extend(predicted_rxns)

        return rxns

    def estimate_num_reactions(self, entries) -> int:
        return comb(len(entries), 2)

    @staticmethod
    def _react_interface(r1, r2, pd, grand_pd=None):
        if grand_pd:
            interface = InterfacialReactivity(
                r1,
                r2,
                grand_pd,
                norm=False,
                include_no_mixing_energy=False,
                pd_non_grand=pd,
                use_hull_energy=True,
            )
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

        entries = pd.all_entries
        rxns = [
            get_computed_rxn(rxn, entries)
            for _, _, _, rxn, _ in interface.get_kinks()
        ]

        return rxns


class MinimizeOpenGibbsEnumerator(MinimizeGibbsEnumerator):
    def __init__(self, open_entries):
        super().__init__()
        self.open_entries = open_entries

    def enumerate(self):
        pass

