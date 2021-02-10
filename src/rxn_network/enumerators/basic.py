from typing import List
from itertools import chain, combinations, compress, groupby, product
from rxn_network.core import Enumerator, Reaction


class BasicEnumerator(Enumerator):
    def __init__(self, entries, n=2):
        self.entries = entries
        self.n = n
        self.combos = combinations(entries, n)

    def enumerate(self, entries) -> List[Reaction]:
        edges = []
        for combo in combos:
            phases = entry.entries
            other_phases = other_entry.entries

            if other_phases == phases:
                continue  # do not consider identity-like reactions (e.g. A + B -> A
                # + B)

            rxn = ComputedReaction(
                list(phases), list(other_phases), num_entries=num_entries
            )
            if not rxn._balanced:
                continue

            if rxn._lowest_num_errors != 0:
                continue  # remove reaction which has components that
                # change sides or disappear

            total_num_atoms = sum(
                [rxn.get_el_amount(elem) for elem in rxn.elements])
            rxn_energy = rxn.calculated_reaction_energy / total_num_atoms

            if rxn_e_filter and rxn_energy > rxn_e_filter:
                continue

            weight = get_rxn_cost(
                rxn, cost_function=cost_function, temp=temp, max_mu_diff=None
            )
            edges.append([v, other_v, weight, rxn, True, False])

        return edges
