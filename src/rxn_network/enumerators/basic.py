from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
from more_itertools import powerset
import numpy as np
from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import get_entry_combinations, get_total_chemsys, group_by_chemsys


class BasicEnumerator(Enumerator):
    def __init__(self, n=2):
        self.n = n

    def enumerate(self, entries, remove_unbalanced=True) -> List[Reaction]:
        combos = get_entry_combinations(entries, self.n)
        combo_dict = group_by_chemsys(combos)

        rxns = []
        for chemsys, selected_combos in combo_dict.items():
            for reactants, products in combinations(selected_combos, 2):
                forward_rxn = ComputedReaction(reactants, products)
                backward_rxn = ComputedReaction(products, reactants)
                rxns.append(forward_rxn)
                rxns.append(backward_rxn)
        if remove_unbalanced:
            rxns = [r for r in rxns if r.balanced]

        return rxns

    def estimate_num_reactions(self, entries) -> int:
        return sum([comb(len(entries), i) for i in range(self.n)])**2
