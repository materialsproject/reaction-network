from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from pymatgen import Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from pymatgen.analysis.interface_reactions import InterfacialReactivity

from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import (
    get_total_chemsys,
    group_by_chemsys,
    filter_entries_by_chemsys,
    get_entry_by_comp,
    get_computed_rxn,
    get_open_computed_rxn,
)


class MinimizeGibbsEnumerator(Enumerator):
    """
    Enumerator for finding all reactions between two reactants that are predicted by
    thermo; i.e., they appear when taking the convex hull along a straight
    line connecting any two phases in G-x phase space. Identity reactions are excluded.
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
                predicted_rxns = self._react_interface(
                    e1.composition, e2.composition, pd
                )
                rxns.extend(predicted_rxns)

        return list(set(rxns))

    def estimate_num_reactions(self, entries) -> int:
        return comb(len(entries), 2)

    @staticmethod
    def _react_interface(r1, r2, pd, grand_pd=None, open_entry=None):
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
            use_original_comps = True
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

            rxns.append(rxn)

        return rxns


class MinimizeGrandPotentialEnumerator(MinimizeGibbsEnumerator):
    """
    Enumerator for finding all reactions between two reactants and an open element
    that are predicted by thermo; i.e., they appear when taking the
    convex hull along a straight line connecting any two phases in Phi-x
    phase space. Identity reactions are excluded.
    """

    def __init__(self, open_elem, chempot):
        super().__init__()
        self.open_elem = Element(open_elem)
        self.chempot = chempot

    def enumerate(self, entries):
        open_entry = sorted(
            filter(lambda e: e.composition.elements == [self.open_elem], entries),
            key=lambda e: e.energy_per_atom,
        )[0]
        entries_no_open = entries.copy()
        entries_no_open.remove(open_entry)
        combos = list(combinations(entries_no_open, 2))
        combos_dict = group_by_chemsys(combos, self.open_elem)

        rxns = []
        for chemsys, combos in combos_dict.items():
            chemsys_entries = filter_entries_by_chemsys(entries, chemsys)
            pd = PhaseDiagram(chemsys_entries)
            grand_pd = GrandPotentialPhaseDiagram(
                entries, {self.open_elem: self.chempot}
            )
            for e1, e2 in combos:
                predicted_rxns = self._react_interface(
                    e1.composition, e2.composition, pd, grand_pd, open_entry
                )
                rxns.extend(predicted_rxns)

        return list(set(rxns))
