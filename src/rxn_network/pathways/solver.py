from tqdm import tqdm
from itertools import chain, combinations, compress, groupby, product
from rxn_network.core import Pathway, Reaction, Solver
from rxn_network.utils import grouper
from rxn_network.pathways.utils import balance_path_arrays
from rxn_network.reactions.computed import ComputedReaction

class PathwaySolver(Solver):
    BATCH_SIZE = 500000
    def __init__(self, entries, reactions, costs):
        super().__init(entries=entries, reactions=reactions, costs=costs)
        self.num_rxns = len(num_rxns)

    def solve(self, net_rxn, max_num_combos=4, find_intermediate_rxns=True):
        self.logger.info(f"Considering {num_rxns} reactions...")

        reactions = self.reactions
        if find_intermediate_rxns:
            reactions.extend(self._find_intermediate_rxns(intermediates))

        total_paths = []
        for n in range(1, max_num_combos + 1):
            if n >= 4:
                self.logger.info(f"Generating and filtering size {n} pathways...")
            all_c_mats, all_m_mats = [], []
            for combos in tqdm(
                grouper(combinations(range(num_rxns), n), self.BATCH_SIZE),
                total=int(comb(num_rxns, n) / self.BATCH_SIZE),
            ):
                comp_matrices = np.stack(
                    [
                        np.vstack([rxn_list[r].vector for r in combo])
                        for combo in combos
                        if combo
                    ]
                )
                c_mats, m_mats = balance_path_arrays(
                    comp_matrices, net_rxn.vector
                )
                all_c_mats.extend(c_mats)
                all_m_mats.extend(m_mats)

            for c_mat, m_mat in zip(all_c_mats, all_m_mats):
                rxn_dict = {}
                for rxn_mat in c_mat:
                    reactant_entries = [
                        self.entries[i]
                        for i in range(len(rxn_mat))
                        if rxn_mat[i] < 0
                    ]
                    product_entries = [
                        self.entries[i]
                        for i in range(len(rxn_mat))
                        if rxn_mat[i] > 0
                    ]
                    rxn = ComputedReaction.balance(
                        reactant_entries, product_entries,
                    )
                    cost = paths_to_all_targets[
                        rxn_list[
                            normalized_rxns.index(
                                Reaction.from_string(rxn.normalized_repr)
                            )
                        ]
                    ]
                    rxn_dict[rxn] = cost
                p = BalancedPathway(rxn_dict, net_rxn, balance=False)
                p.set_multiplicities(m_mat.flatten())
                total_paths.append(p)

    def _find_intermediate_rxns(self, intermediates):
        pass
