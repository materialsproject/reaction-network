"""
Implements a reaction pathway solver class which efficiently solves mass balance
equations using matrix operations.
"""

from copy import deepcopy
from itertools import combinations
from math import comb
from typing import Union

import numpy as np
import ray
from numba import njit, prange
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.core.solver import Solver
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.pathways.balanced import BalancedPathway
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import grouper
from rxn_network.utils.ray import initialize_ray, to_iterator


class PathwaySolver(Solver):
    """
    Solver that implements an efficient method (using numba) for finding balanced
    reaction pathways from a list of graph-derived reaction pathways (i.e. a list of
    lists of reactions)
    """

    def __init__(
        self,
        pathways: PathwaySet,
        entries: GibbsEntrySet,
        cost_function: CostFunction,
        open_elem: str = None,
        chempot: float = None,
        chunk_size=100000,
        batch_size=None,
    ):
        """
        Args:
            pathways: List of reaction pathways derived from the network.
            entries: GibbsEntrySet containing all entries in the network.
            cost_function: CostFunction object to use for the solver.
            open_elem: Element to use for pathways with an open element.
            chempot: Chemical potential to use for pathways with an open element.
        """
        super().__init__(pathways=deepcopy(pathways))
        self._entries = entries
        self.cost_function = cost_function
        self.open_elem = Element(open_elem) if open_elem else None
        self.chempot = chempot
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def solve(
        self,
        net_rxn: Union[ComputedReaction, OpenComputedReaction],
        max_num_combos: int = 4,
        find_intermediate_rxns: bool = True,
        intermediate_rxn_energy_cutoff: float = 0.0,
        use_basic_enumerator: bool = True,
        use_minimize_enumerator: bool = False,
        filter_interdependent: bool = True,
    ) -> PathwaySet:
        """

        Args:
            net_rxn: The reaction representing the total reaction from precursors to
                final targets.
            max_num_combos: The maximum allowable size of the balanced reaction pathway.
                At values <=5, the solver will start to take a significant amount of
                time to run.
            find_intermediate_rxns: Whether to find intermediate reactions; crucial for
                finding pathways where intermediates react together, as these reactions
                may not occur in the graph-derived pathways. Defaults to True.
            intermediate_rxn_energy_cutoff: An energy cutoff by which to filter down
                intermediate reactions. This can be useful when there are a large number
                of possible intermediates. < 0 means allow only exergonic reactions.
            use_basic_enumerator: Whether to use the BasicEnumerator to find
                intermediate reactions. Defaults to True.
            use_minimize_enumerator: Whether to use the MinimizeGibbsEnumerator to find
                intermediate reactions. Defaults to False.
            filter_interdependent: Whether or not to filter out pathways where reaction
                steps are interdependent. Defaults to True.

        Returns:
            A list of BalancedPathway objects.
        """

        if not net_rxn.balanced:
            raise ValueError(
                "Net reaction must be balanceable to find all reaction pathways."
            )

        initialize_ray()

        entries_copy = deepcopy(self.entries)
        entries = entries_copy.entries_list
        num_entries = len(entries)

        reactions = deepcopy(self.reactions)
        costs = deepcopy(self.costs)

        precursors = deepcopy(net_rxn.reactant_entries)
        targets = deepcopy(net_rxn.product_entries)

        self.logger.info(f"Net reaction: {net_rxn} \n")

        if find_intermediate_rxns:
            self.logger.info("Identifying reactions between intermediates...")

            intermediate_rxns = self._find_intermediate_rxns(
                targets,
                intermediate_rxn_energy_cutoff,
                use_basic_enumerator,
                use_minimize_enumerator,
            )
            intermediate_costs = [
                self.cost_function.evaluate(r) for r in intermediate_rxns.get_rxns()
            ]
            for r, c in zip(intermediate_rxns, intermediate_costs):
                if r not in reactions:
                    reactions.append(r)
                    costs.append(c)

        net_rxn_vector = net_rxn.get_entry_idx_vector(num_entries)

        if net_rxn in reactions:
            reactions.remove(net_rxn)

        reaction_set = ray.put(ReactionSet.from_rxns(reactions))
        entries = ray.put(entries)
        costs = ray.put(costs)
        num_entries = ray.put(num_entries)
        net_rxn_vector = ray.put(net_rxn_vector)
        open_elem = ray.put(self.open_elem)
        chempot = ray.put(self.chempot)

        num_rxns = len(reactions)
        batch_size = self.batch_size or ray.cluster_resources()["CPU"] - 1

        num_combos = sum(comb(num_rxns, k) for k in range(1, max_num_combos + 1))
        num_batches = int((num_combos // self.chunk_size + 1) // batch_size + 1)

        paths = []
        paths_refs = []
        batch_count = 1
        for n in range(1, max_num_combos + 1):
            for group in grouper(combinations(range(num_rxns), n), self.chunk_size):
                paths_refs.append(
                    _get_balanced_paths_ray.remote(
                        group,
                        reaction_set,
                        costs,
                        entries,
                        num_entries,
                        net_rxn_vector,
                        open_elem,
                        chempot,
                    )
                )
                if len(paths_refs) >= batch_size:
                    for paths_ref in tqdm(
                        to_iterator(paths_refs),
                        total=len(paths_refs),
                        desc=(
                            f"{self.__class__.__name__} (Batch"
                            f" {batch_count}/{num_batches})"
                        ),
                    ):
                        paths.extend(paths_ref)

                    batch_count += 1

                    paths_refs = []

        for paths_ref in tqdm(
            to_iterator(paths_refs),
            total=len(paths_refs),
            desc=f"{self.__class__.__name__} (Batch {batch_count}/{num_batches})",
        ):
            paths.extend(paths_ref)

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

        return PathwaySet.from_paths(filtered_paths)

    def _find_intermediate_rxns(
        self,
        targets,
        energy_cutoff,
        use_basic_enumerator,
        use_minimize_enumerator,
    ):
        """
        Method for finding intermediate reactions using enumerators and
        specified settings.
        """

        intermediates = {e for rxn in self.reactions for e in rxn.entries}
        intermediates = GibbsEntrySet(
            list(intermediates) + targets,
        )
        target_formulas = [e.composition.reduced_formula for e in targets]
        ref_elems = {e for e in self.entries if e.is_element}

        intermediates = intermediates | ref_elems

        rxn_set = ReactionSet(
            intermediates.entries_list,
            [],
            [],
            open_elem=self.open_elem,
            chempot=self.chempot,
            all_data=[],
        )

        if use_basic_enumerator:
            be = BasicEnumerator(targets=target_formulas, calculate_e_above_hulls=False)
            rxn_set = rxn_set.add_rxn_set(be.enumerate(intermediates))

            if self.open_elem:
                boe = BasicOpenEnumerator(
                    open_phases=[Composition(str(self.open_elem)).reduced_formula],
                    targets=target_formulas,
                    calculate_e_above_hulls=False,
                )

                rxn_set = rxn_set.add_rxn_set(boe.enumerate(intermediates))

        if use_minimize_enumerator:
            mge = MinimizeGibbsEnumerator(
                targets=target_formulas, calculate_e_above_hulls=False
            )
            rxn_set = rxn_set.add_rxn_set(mge.enumerate(intermediates))

            if self.open_elem:
                mgpe = MinimizeGrandPotentialEnumerator(
                    open_elem=self.open_elem,
                    mu=self.chempot,
                    targets=target_formulas,
                )
                rxn_set.add_rxn_set(mgpe.enumerate(intermediates))

        rxns = list(filter(lambda x: x.energy_per_atom < energy_cutoff, rxn_set))
        rxns = [r for r in rxns if all(e in intermediates for e in r.entries)]
        num_rxns = len(rxns)
        rxns = ReactionSet.from_rxns(rxns, filter_duplicates=True)

        self.logger.info(f"Found {num_rxns} intermediate reactions! \n")

        return rxns

    @property
    def entries(self) -> GibbsEntrySet:
        """Entry set used in solver"""
        return self._entries


@njit(parallel=True, fastmath=True)
def _balance_path_arrays(
    comp_matrices: np.ndarray,
    net_coeffs: np.ndarray,
    tol: float = 1e-6,
):
    """
    Fast solution for reaction multiplicities via mass balance stochiometric
    constraints. Parallelized using Numba. Can be applied to large batches (100K-1M
    sets of reactions at a time.)

    Args:
        comp_matrices: Array containing stoichiometric coefficients of all
            compositions in all reactions, for each trial combination.
        net_coeffs: Array containing stoichiometric coefficients of net reaction.
        tol: numerical tolerance for determining if a multiplicity is zero
            (reaction was removed).
    """
    shape = comp_matrices.shape
    net_coeff_filter = np.argwhere(net_coeffs != 0).flatten()
    len_net_coeff_filter = len(net_coeff_filter)
    all_multiplicities = np.zeros((shape[0], shape[1]), np.float64)
    indices = np.full(shape[0], False)

    for i in prange(shape[0]):  # pylint: disable=not-an-iterable
        correct = True
        for j in range(len_net_coeff_filter):
            idx = net_coeff_filter[j]
            if not comp_matrices[i][:, idx].any():
                correct = False
                break
        if not correct:
            continue

        comp_pinv = np.linalg.pinv(comp_matrices[i]).T
        multiplicities = comp_pinv @ net_coeffs
        solved_coeffs = comp_matrices[i].T @ multiplicities

        if (multiplicities < tol).any():
            continue

        if not (
            np.abs(solved_coeffs - net_coeffs) <= (1e-08 + 1e-05 * np.abs(net_coeffs))
        ).all():
            continue

        all_multiplicities[i] = multiplicities
        indices[i] = True

    filtered_indices = np.argwhere(indices != 0).flatten()
    length = filtered_indices.shape[0]
    filtered_comp_matrices = np.empty((length, shape[1], shape[2]), np.float64)
    filtered_multiplicities = np.empty((length, shape[1]), np.float64)

    for i in range(length):
        idx = filtered_indices[i]
        filtered_comp_matrices[i] = comp_matrices[idx]
        filtered_multiplicities[i] = all_multiplicities[idx]

    return filtered_comp_matrices, filtered_multiplicities


def _create_comp_matrices(combos, rxns, num_entries):
    """Create array of stoichiometric coefficients for each reaction."""
    comp_matrices = np.stack(
        [
            np.vstack([rxns[r].get_entry_idx_vector(num_entries) for r in combo])
            for combo in combos
            if combo
        ]
    )
    return comp_matrices


@ray.remote
def _get_balanced_paths_ray(
    combos,
    reaction_set,
    costs,
    entries,
    num_entries,
    net_rxn_vector,
    open_elem,
    chempot,
):
    reactions = list(reaction_set.get_rxns())
    comp_matrices = _create_comp_matrices(combos, reactions, num_entries)

    paths = []

    c_mats, m_mats = _balance_path_arrays(comp_matrices, net_rxn_vector)

    for c_mat, m_mat in zip(c_mats, m_mats):
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

            if open_elem is not None:
                rxn = OpenComputedReaction(
                    entries=ents,
                    coefficients=coeffs,
                    chempots={open_elem: chempot},
                )

            else:
                rxn = ComputedReaction(entries=ents, coefficients=coeffs)

            try:
                path_rxns.append(rxn)
                path_costs.append(costs[reactions.index(rxn)])
            except Exception as e:
                print(e)
                continue

        p = BalancedPathway(path_rxns, m_mat.flatten(), path_costs, balanced=True)
        paths.append(p)

    return paths
