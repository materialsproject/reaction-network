"""
Implements a reaction pathway solver class which efficiently solves mass balance
equations using matrix operations.
"""

from itertools import combinations
from typing import List
from copy import deepcopy

import numpy as np
from scipy.special import comb
from tqdm.notebook import tqdm

from pymatgen.core.composition import Composition

from rxn_network.core import CostFunction, Pathway, Solver
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.pathways.balanced import BalancedPathway
from rxn_network.pathways.utils import balance_path_arrays
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils import grouper


class PathwaySolver(Solver):
    """
    Solver that implements an efficient numba method for finding balanced
    reaction pathways from a list of graph-derived reaction pathways (i.e. a list of lists of reactions)
    """

    BATCH_SIZE = 500000

    def __init__(
        self,
        entries: GibbsEntrySet,
        pathways: List[Pathway],
        cost_function: CostFunction,
        open_elem: str = None,
        chempot: float = None,
    ):
        """

        Args:
            entries: GibbsEntrySet containing all entries in the network.
            pathways: List of reaction pathways derived from the network.
            cost_function: CostFunction object to use for the solver.
            open_elem: Element to use for pathways with an open element.
            chempot: Chemical potential to use for pathways with an open element.
        """
        super().__init__(entries=deepcopy(entries), pathways=deepcopy(pathways))
        self.cost_function = cost_function
        self.open_elem = open_elem
        self.chempot = chempot

    def solve(
        self,
        net_rxn: ComputedReaction,
        max_num_combos: int = 4,
        find_intermediate_rxns: bool = True,
        intermediate_rxn_energy_cutoff: float = 0.0,
        use_basic_enumerator: bool = True,
        use_minimize_enumerator: bool = False,
        filter_interdependent: bool = True,
    ):
        """

        Args:
            net_rxn: The reaction representing the total reaction from precursors to
                final targets.
            max_num_combos: The maximum allowable size of the balanced reaction pathway.
                At 5 or more, the solver will be very slow.
            find_intermediate_rxns: Whether to find intermediate reactions; crucial for
                finding pathways where intermediates react together, as these reactions may
                not occur in the graph-derived pathways. Defaults to True.
            intermediate_rxn_energy_cutoff: An energy cutoff by which to filter down
                intermediate reactions. This can be useful when there are a large number of
                possible intermediates. < 0 means allow only exergonic reactions.
            use_basic_enumerator: Whether to use the BasicEnumerator to find intermediate
                reactions. Defaults to True.
            use_minimize_enumerator: Whether to use the MinimizeGibbsEnumerator to find
                intermediate reactions. Defaults to False.
            filter_interdependent: Whether or not to filter out pathways where reaction
                steps are interdependent. Defaults to True.

        Returns:
            A list of BalancedPathway objects.
        """
        entries = deepcopy(self.entries)
        entries = entries.entries_list
        num_entries = len(entries)

        reactions = deepcopy(self.reactions)
        costs = deepcopy(self.costs)

        precursors = deepcopy(net_rxn.reactant_entries)
        targets = deepcopy(net_rxn.product_entries)

        if not net_rxn.balanced:
            raise ValueError(
                "Net reaction must be balanceable to find all reaction pathways."
            )

        self.logger.info(f"NET RXN: {net_rxn} \n")

        if find_intermediate_rxns:
            self.logger.info("Identifying reactions between intermediates...")

            intermediate_rxns = self._find_intermediate_rxns(
                precursors,
                targets,
                intermediate_rxn_energy_cutoff,
                use_basic_enumerator,
                use_minimize_enumerator,
            )
            intermediate_costs = [
                self.cost_function.evaluate(r) for r in intermediate_rxns
            ]
            for r, c in zip(intermediate_rxns, intermediate_costs):
                if r not in reactions:
                    reactions.append(r)
                    costs.append(c)

        for r in reactions:
            v = self._build_idx_vector(r, num_entries)

        net_rxn_vector = self._build_idx_vector(net_rxn, num_entries)
        if net_rxn in reactions:
            reactions.remove(net_rxn)

        paths = []
        for n in range(1, max_num_combos + 1):
            total = int(comb(len(reactions), n) / self.BATCH_SIZE) + 1
            groups = grouper(combinations(range(len(reactions)), n), self.BATCH_SIZE)

            pbar = groups
            if n >= 4:
                self.logger.info(f"Solving for balanced pathways of size {n} ...")
                pbar = tqdm(groups, total=total, desc="PathwaySolver")

            all_c_mats, all_m_mats = [], []
            for idx, combos in enumerate(pbar):
                if n >= 4:
                    pbar.set_description(
                        f"{self.BATCH_SIZE*idx}/{total*self.BATCH_SIZE}"
                    )
                comp_matrices = np.stack(
                    [
                        np.vstack(
                            [
                                self._build_idx_vector(reactions[r], num_entries)
                                for r in combo
                            ]
                        )
                        for combo in combos
                        if combo
                    ]
                )
                c_mats, m_mats = balance_path_arrays(comp_matrices, net_rxn_vector)
                all_c_mats.extend(c_mats)
                all_m_mats.extend(m_mats)

            for c_mat, m_mat in zip(all_c_mats, all_m_mats):
                path_rxns = []
                path_costs = []
                for rxn_mat in c_mat:
                    ents, coeffs = zip(
                        *[
                            (entries[idx], c)
                            for idx, c in enumerate(rxn_mat)
                            if not np.isclose(c, 0.0)
                        ]
                    )

                    rxn = ComputedReaction(entries=ents, coefficients=coeffs)

                    try:
                        path_rxns.append(rxn)
                        path_costs.append(costs[reactions.index(rxn)])
                    except Exception as e:
                        print(e)
                        continue

                p = BalancedPathway(
                    path_rxns, m_mat.flatten(), path_costs, balanced=True
                )
                paths.append(p)

        filtered_paths = []
        if filter_interdependent:
            precursor_comps = [p.composition for p in precursors]
            for p in paths:
                interdependent = p.contains_interdependent_rxns(precursor_comps)
                if not interdependent:
                    filtered_paths.append(p)
        else:
            filtered_paths = paths

        filtered_paths = sorted(list(set(filtered_paths)), key=lambda p: p.average_cost)
        return filtered_paths

    @staticmethod
    def _build_idx_vector(rxn, num_entries):
        """
        Builds a vector of indices for a reaction.

        Args:
            rxn:

        Returns:

        """
        indices = [e.data.get("idx") for e in rxn.entries]
        v = np.zeros(num_entries)
        v[indices] = rxn.coefficients
        return v

    def _find_intermediate_rxns(
        self,
        precursors,
        targets,
        energy_cutoff,
        use_basic_enumerator,
        use_minimize_enumerator,
    ):
        rxns = []

        intermediates = {e for rxn in self.reactions for e in rxn.entries}
        intermediates = GibbsEntrySet(
            list(intermediates) + targets,
        )
        target_formulas = [e.composition.reduced_formula for e in targets]
        ref_elems = {e for e in self.entries if e.is_element}

        if use_basic_enumerator:
            be = BasicEnumerator(
                targets=target_formulas,
            )
            rxns.extend(be.enumerate(intermediates))

            if self.open_elem:
                boe = BasicOpenEnumerator(
                    open_phases=[Composition(str(self.open_elem)).reduced_formula],
                    targets=target_formulas,
                )

                rxns.extend(boe.enumerate(intermediates))

        if use_minimize_enumerator:
            ents = deepcopy(intermediates)
            ents = ents | ref_elems

            mge = MinimizeGibbsEnumerator(targets=target_formulas)
            rxns.extend(mge.enumerate(ents))

            if self.open_elem:
                mgpe = MinimizeGrandPotentialEnumerator(
                    open_elem=self.open_elem, mu=self.chempot, targets=target_formulas
                )
                rxns.extend(mgpe.enumerate(ents))

        rxns = list(filter(lambda x: x.energy_per_atom < energy_cutoff, rxns))
        rxns = [r for r in rxns if all([e in intermediates for e in r.entries])]
        rxns = [r for r in rxns if (len(r.reactants) < 4 and len(r.products) < 4)]

        self.logger.info(f"Found {len(rxns)} intermediate reactions!")

        return rxns
