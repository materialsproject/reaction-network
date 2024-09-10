"""Implements a reaction pathway solver class which efficiently solves mass balance
equations using matrix operations.
"""

from __future__ import annotations

from abc import ABCMeta
from copy import deepcopy
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import ray
from monty.json import MSONable
from numba import jit
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core import Composition
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
from rxn_network.utils.funcs import get_logger, grouper
from rxn_network.utils.ray import initialize_ray, to_iterator

if TYPE_CHECKING:
    from rxn_network.costs.base import CostFunction
    from rxn_network.pathways.base import Pathway
    from rxn_network.reactions.base import Reaction

logger = get_logger(__name__)


class Solver(MSONable, metaclass=ABCMeta):
    """Base definition for a pathway solver class."""

    def __init__(self, pathways: PathwaySet):
        """
        Args:
            pathways: A PathwaySet object containing the pathways to be combined/balanced.
        """
        self._pathways = pathways

        rxns = []
        costs = []

        for path in self._pathways.paths:
            for rxn, cost in zip(path.reactions, path.costs):
                if rxn not in rxns:
                    rxns.append(rxn)
                    costs.append(cost)

        self._reactions = rxns
        self._costs = costs

    @property
    def pathways(self) -> list[Pathway]:
        """Pathways used in solver class."""
        return self._pathways

    @property
    def reactions(self) -> list[Reaction]:
        """Reactions used in solver class."""
        return self._reactions

    @property
    def costs(self) -> list[float]:
        """Costs used in solver class."""
        return self._costs

    @property
    def num_rxns(self) -> int:
        """Length of the reaction list."""
        return len(self.reactions)

    @property
    def num_entries(self) -> int:
        """Length of entry list."""
        return len(self._entries)


class PathwaySolver(Solver):
    """Solver that implements an efficient method (using numba) for finding balanced
    reaction pathways from a list of graph-derived reaction pathways (i.e. a list of
    lists of reactions).

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.
    """

    def __init__(
        self,
        pathways: PathwaySet,
        entries: GibbsEntrySet,
        cost_function: CostFunction,
        open_elem: str | Element | None = None,
        chempot: float = 0.0,
        chunk_size: int = 100000,
        batch_size: int | None = None,
    ):
        """
        Args:
            pathways: List of reaction pathways derived from the network.
            entries: GibbsEntrySet containing all entries in the network.
            cost_function: CostFunction object to use for the solver.
            open_elem: Optional element to use for pathways with an open element.
            chempot: Chemical potential to use for pathways with an open element.
                Defaults to 0.0.
            chunk_size: The number of pathways per chunk to use for balancing. Defaults
                to 100,000.
            batch_size: Number of chunks to submit to each CPU at a time. Automatically
                calculated if not set.


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
        net_rxn: ComputedReaction | OpenComputedReaction,
        max_num_combos: int = 4,
        find_intermediate_rxns: bool = True,
        intermediate_rxn_energy_cutoff: float = 0.0,
        use_basic_enumerator: bool = True,
        use_minimize_enumerator: bool = False,
        filter_interdependent: bool = True,
    ) -> PathwaySet:
        """Args:
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
            raise ValueError("Net reaction must be balanceable to find all reaction pathways.")

        if not ray.is_initialized():
            initialize_ray()

        entries_copy = deepcopy(self.entries)
        entries = entries_copy.entries_list
        num_entries = len(entries)

        reactions = deepcopy(self.reactions)
        costs = deepcopy(self.costs)

        precursors = deepcopy(net_rxn.reactant_entries)
        targets = deepcopy(net_rxn.product_entries)

        logger.info(f"Net reaction: {net_rxn} \n")

        if find_intermediate_rxns:
            logger.info("Identifying reactions between intermediates...")

            intermediate_rxns = self._find_intermediate_rxns(
                targets,
                intermediate_rxn_energy_cutoff,
                use_basic_enumerator,
                use_minimize_enumerator,
            )
            intermediate_costs = [self.cost_function.evaluate(r) for r in intermediate_rxns.get_rxns()]
            for r, c in zip(intermediate_rxns, intermediate_costs):
                if r not in reactions:
                    reactions.append(r)
                    costs.append(c)

        clean_r_set = ReactionSet.from_rxns(reactions, filter_duplicates=True)
        cleaned_reactions, cleaned_costs = zip(
            *[(r, c) for r, c in zip(reactions, costs) if r in clean_r_set and r != net_rxn]
        )

        net_rxn_vector = net_rxn.get_entry_idx_vector(num_entries)

        num_rxns = len(cleaned_reactions)
        num_cpus = ray.cluster_resources()["CPU"]
        batch_size = self.batch_size or num_cpus - 1

        net_coeff_filter = np.argwhere(net_rxn_vector != 0).flatten()
        net_coeff_filter = ray.put(net_coeff_filter)
        cleaned_reactions_ref = ray.put(cleaned_reactions)

        comp_matrices = {n: [] for n in range(1, max_num_combos + 1)}  # type: ignore
        comp_matrices_refs_dict = {}  # type: ignore

        for n in range(1, max_num_combos + 1):
            comp_matrices_refs_dict[n] = []
            for group in grouper(combinations(range(num_rxns), n), self.chunk_size):
                comp_matrices_refs_dict[n].append(
                    _create_comp_matrices.remote(group, cleaned_reactions_ref, num_entries, net_coeff_filter)
                )

        logger.info("Building comp matrices...")

        num_objs = sum(len(i) for i in comp_matrices_refs_dict.values())  # type: ignore

        with tqdm(total=num_objs) as pbar:
            for n, comp_matrices_refs in comp_matrices_refs_dict.items():
                for comp_matrices_ref in to_iterator(comp_matrices_refs):
                    pbar.update(1)
                    comp_matrices[n].append(comp_matrices_ref)

                comp_matrices[n] = np.concatenate(comp_matrices[n])

                if not comp_matrices[n].any():  # type: ignore
                    del comp_matrices[n]

        logger.info("Comp matrices done...")

        num_cpu_jobs = 0

        c_m_mats = []
        c_m_mats_refs = []

        num_jobs = sum(len(val) // self.chunk_size + 1 for val in comp_matrices.values())
        num_batches = int(num_jobs // batch_size + 1)

        batch_count = 1

        for n, comp_matrix in comp_matrices.items():
            if n >= 4:
                num_splits = len(comp_matrix) // self.chunk_size + 1
                splits = np.array_split(comp_matrix, num_splits)
            else:
                splits = [comp_matrix]  # only submit one job for small n

            for group in splits:
                if len(group) == 0:  # catch empty matrices
                    continue

                path_balancer = _balance_path_arrays_cpu_wrapper
                num_cpu_jobs += 1

                c_m_mats_refs.append(
                    path_balancer.remote(
                        group,
                        net_rxn_vector,
                    )
                )

                if len(c_m_mats_refs) >= batch_size:
                    for c_m_mats_ref in tqdm(
                        to_iterator(c_m_mats_refs),
                        total=len(c_m_mats_refs),
                        desc=(f"{self.__class__.__name__} (Batch {batch_count}/{num_batches})"),
                    ):
                        c_m_mats.append(c_m_mats_ref)  # noqa: PERF402

                    batch_count += 1

                    num_cpu_jobs = 0

                    c_m_mats_refs = []

        for c_m_mats_ref in tqdm(
            to_iterator(c_m_mats_refs),
            total=len(c_m_mats_refs),
            desc=f"{self.__class__.__name__} (Batch {batch_count}/{num_batches})",
        ):
            c_m_mats.append(c_m_mats_ref)  # noqa: PERF402

        c_mats, m_mats = zip(*c_m_mats)
        c_mats = [mat for mats in c_mats for mat in mats if mat is not None]  # type: ignore
        m_mats = [mat for mats in m_mats for mat in mats if mat is not None]  # type: ignore

        paths = []
        for c_mat, m_mat in zip(c_mats, m_mats):
            path_rxns = []
            path_costs = []

            for rxn_mat in c_mat:
                ents, coeffs = zip(*[(entries[idx], c) for idx, c in enumerate(rxn_mat) if not np.isclose(c, 0.0)])

                if self.open_elem is not None:
                    rxn = OpenComputedReaction(
                        entries=ents,
                        coefficients=coeffs,
                        chempots={self.open_elem: self.chempot},
                    )

                else:
                    rxn = ComputedReaction(entries=ents, coefficients=coeffs)

                try:
                    path_rxns.append(rxn)
                    path_costs.append(cleaned_costs[cleaned_reactions.index(rxn)])
                except Exception as e:
                    print(e)
                    continue

            p = BalancedPathway(path_rxns, m_mat.flatten(), path_costs, balanced=True)
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

        filtered_paths = sorted(set(filtered_paths), key=lambda p: p.average_cost)

        return PathwaySet.from_paths(filtered_paths)

    def _find_intermediate_rxns(
        self,
        targets,
        energy_cutoff,
        use_basic_enumerator,
        use_minimize_enumerator,
    ):
        """Method for finding intermediate reactions using enumerators and
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
            {},
            {},
            open_elem=self.open_elem,
            chempot=self.chempot,
            all_data={},
        )

        if use_basic_enumerator:
            be = BasicEnumerator(targets=target_formulas)
            rxn_set = rxn_set.add_rxn_set(be.enumerate(intermediates))

            if self.open_elem:
                boe = BasicOpenEnumerator(
                    open_phases=[Composition(str(self.open_elem)).reduced_formula],
                    targets=target_formulas,
                )

                rxn_set = rxn_set.add_rxn_set(boe.enumerate(intermediates))

        if use_minimize_enumerator:
            mge = MinimizeGibbsEnumerator(
                targets=target_formulas,
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

        logger.info(f"Found {num_rxns} intermediate reactions! \n")

        return rxns

    @property
    def entries(self) -> GibbsEntrySet:
        """Entry set used in solver."""
        return self._entries


@jit(nopython=True)
def _balance_path_arrays_cpu(
    comp_matrices: np.ndarray,
    net_coeffs: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast solution for reaction multiplicities via mass balance stochiometric
    constraints. Parallelized using Numba JIT. Can be applied to large batches (100K-1M
    sets of reactions at a time.).

    Args:
        comp_matrices: Array containing stoichiometric coefficients of all
            compositions in all reactions, for each trial combination.
        net_coeffs: Array containing stoichiometric coefficients of net reaction.
        tol: numerical tolerance for determining if a multiplicity is zero
            (i.e., if reaction was removed).
    """
    shape = comp_matrices.shape
    net_coeff_filter = np.argwhere(net_coeffs != 0).flatten()
    len_net_coeff_filter = len(net_coeff_filter)
    all_multiplicities = np.zeros((shape[0], shape[1]), np.float64)
    indices = np.full(shape[0], fill_value=False)

    for i in range(shape[0]):
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

        if not (np.abs(solved_coeffs - net_coeffs) <= (1e-08 + 1e-05 * np.abs(net_coeffs))).all():
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


@ray.remote
def _create_comp_matrices(combos, rxns, num_entries, net_coeff_filter):
    """Create array of stoichiometric coefficients for each reaction."""
    comp_matrices = np.stack(
        [np.vstack([rxns[r].get_entry_idx_vector(num_entries) for r in combo]) for combo in combos if combo]
    )
    # filter bad matrices
    return comp_matrices[comp_matrices[:, :, net_coeff_filter].any(axis=1).all(axis=1)]


@ray.remote
def _balance_path_arrays_cpu_wrapper(
    comp_matrices,
    net_rxn_vector,
):
    """Wraps pathway balancing method with ray.remote decorator."""
    return _balance_path_arrays_cpu(comp_matrices, net_rxn_vector)
