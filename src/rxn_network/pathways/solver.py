from tqdm.notebook import tqdm
import numpy as np
from itertools import chain, combinations, compress, groupby, product
from rxn_network.core import Pathway, Reaction, Solver
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.utils import grouper
from rxn_network.pathways.utils import balance_path_arrays
from rxn_network.pathways.balanced import BalancedPathway
from rxn_network.reactions.computed import ComputedReaction
from scipy.special import comb


class PathwaySolver(Solver):
    BATCH_SIZE = 500000

    def __init__(self, entries, pathways, cost_function):

        super().__init__(entries=entries, pathways=pathways)
        self.cost_function = cost_function

    def solve(
        self,
        net_rxn,
        max_num_combos=4,
        find_intermediate_rxns=True,
        filter_interdependent=True,
    ):
        entries = list(self.entries)
        precursors = net_rxn.reactant_entries
        targets = net_rxn.product_entries

        reactions = self.reactions.copy()
        costs = self.costs.copy()

        if find_intermediate_rxns:
            self.logger.info("Identifying reactions between intermediates...")
            intermediate_rxns = self._find_intermediate_rxns(precursors, targets)
            intermediate_costs = [
                self.cost_function.evaluate(r) for r in intermediate_rxns
            ]
            for r, c in zip(intermediate_rxns, intermediate_costs):
                if r not in reactions:
                    reactions.append(r)
                    costs.append(c)

        paths = []
        for n in range(1, max_num_combos + 1):
            total = int(comb(len(reactions), n) / self.BATCH_SIZE) + 1
            groups = grouper(combinations(range(len(reactions)), n), self.BATCH_SIZE)

            pbar = groups
            if n >= 4:
                self.logger.info(f"Generating and filtering size {n} pathways...")
                pbar = tqdm(groups, total=total, desc="PathwaySolver")

            all_c_mats, all_m_mats = [], []
            for idx, combos in enumerate(pbar):
                if n >= 4:
                    pbar.set_description(
                        f"{self.BATCH_SIZE*idx}/{total*self.BATCH_SIZE}"
                    )
                comp_matrices = np.stack(
                    [
                        np.vstack([self.build_idx_vector(reactions[r]) for r in combo])
                        for combo in combos
                        if combo
                    ]
                )
                c_mats, m_mats = balance_path_arrays(
                    comp_matrices, self.build_idx_vector(net_rxn)
                )
                all_c_mats.extend(c_mats)
                all_m_mats.extend(m_mats)

            for c_mat, m_mat in zip(all_c_mats, all_m_mats):
                path_rxns = []
                path_costs = []
                for rxn_mat in c_mat:
                    reactant_entries = [
                        entries[i] for i in range(len(rxn_mat)) if rxn_mat[i] < 0
                    ]
                    product_entries = [
                        entries[i] for i in range(len(rxn_mat)) if rxn_mat[i] > 0
                    ]
                    rxn = ComputedReaction.balance(
                        reactant_entries,
                        product_entries,
                    )
                    path_rxns.append(rxn)
                    path_costs.append(costs[reactions.index(rxn)])

                p = BalancedPathway(
                    path_rxns, m_mat.flatten(), path_costs, balanced=True
                )
                paths.append(p)

        filtered_paths = []
        if filter_interdependent:
            precursor_comps = [p.composition for p in precursors]
            for p in paths:
                interdependent, _ = p.contains_interdependent_rxns(precursor_comps)
                if not interdependent:
                    filtered_paths.append(p)
        else:
            filtered_paths = paths

        return sorted(list(set(filtered_paths)), key=lambda p: p.average_cost)

    def _find_intermediate_rxns(self, precursors, targets):
        rxns = []
        intermediates = {e for rxn in self.reactions for e in rxn.entries}
        intermediates = intermediates - set(precursors) - set(targets)
        intermediate_formulas = [e.composition.reduced_formula for e in intermediates]

        if not targets:
            be = BasicEnumerator(precursors=intermediate_formulas)
            rxns = be.enumerate(self.entries)
        else:
            for target in targets:
                be = BasicEnumerator(
                    precursors=intermediate_formulas,
                    target=target.composition.reduced_formula,
                )
                rxns.extend(be.enumerate(self.entries))

        return rxns

    def build_idx_vector(self, rxn):
        indices = [e.data.get("idx") for e in rxn.entries]
        v = np.zeros(self.num_entries)
        v[indices] = rxn.coefficients
        return v
