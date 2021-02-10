from typing import List
from itertools import chain, combinations, compress, groupby, product
from math import comb
import numpy as np
from rxn_network.core import Enumerator, Reaction
from rxn_network.reactions import ComputedReaction
from rxn_network.enumerators.utils import get_entry_combinations, get_total_chemsys, group_by_chemsys


class BasicEnumerator(Enumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n).
    """
    def __init__(self, n):
        self.n = n

    def enumerate(self, entries, remove_unbalanced=True, remove_changed=True) -> List[
        Reaction]:

        combos = get_entry_combinations(entries, self.n)
        combo_dict = group_by_chemsys(combos)

        rxns = []
        for chemsys, selected_combos in combo_dict.items():
            for reactants, products in combinations(selected_combos, 2):
                forward_rxn = ComputedReaction.balance(reactants, products)
                if remove_unbalanced and not (forward_rxn.balanced):
                    continue
                if remove_changed and forward_rxn.lowest_num_errors != 0:
                    continue
                backward_rxn = ComputedReaction(products, reactants,
                                                -1*forward_rxn.coefficients)
                rxns.append(forward_rxn)
                rxns.append(backward_rxn)
        return rxns

    def estimate_num_reactions(self, entries) -> int:
        return sum([comb(len(entries), i) for i in range(self.n)])**2


class BasicOpenEnumerator(BasicEnumerator):
    """
    Enumerator for finding all simple reactions within a set of entries, up to a
    maximum reactant/product cardinality (n), with any number of open elements.
    """
    def __init__(self, n, open_elements):
        super().__init()

