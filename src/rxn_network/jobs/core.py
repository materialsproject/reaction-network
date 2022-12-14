"""Core jobs for reaction-network creation and analysis."""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import ray
from jobflow import SETTINGS, Maker, job
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.costs.calculators import (
    ChempotDistanceCalculator,
    PrimarySelectivityCalculator,
    SecondarySelectivityAreaCalculator,
    SecondarySelectivityCalculator,
    SecondarySelectivityMaxCalculator,
)
from rxn_network.costs.softplus import Softplus
from rxn_network.entries.utils import get_all_entries_in_chemsys, process_entries
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.jobs.schema import (
    EntrySetDocument,
    EnumeratorTaskDocument,
    NetworkTaskDocument,
    PathwaySolverTaskDocument,
    SelectivitiesTaskDocument,
)
from rxn_network.jobs.utils import get_added_elem_data, run_enumerators
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.basic import BasicReaction
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger, grouper
from rxn_network.utils.ray import initialize_ray

logger = get_logger(__name__)


@dataclass
class GetEntrySetMaker(Maker):
    """
    Maker to create job for acquiring and processing entries to be used in reaction
    enumeration or network building.

    Args (A list of argumnets provided to the dataclass)
        name: Name of the job.
        entry_db_name: Name of the entry database store to use. If none is available,
            will automatically use MPRester to acquire entries.
        temperature: Temperature to use for computing thermodynamic properties.
        include_nist_data: Whether to include NIST data in the entry set.
        include_barin_data: Whether to include Barin data in the entry set.
        include_freed_data: Whether to include FREED data in the entry set.
        e_above_hull: Energy above hull to use for filtering entries.
        include_polymorphs: Whether to include polymorphs in the entry set.
        formulas_to_include: List of formulas to include in the entry set.
        calculate_e_above_hulls: Whether to calculate e_above_hulls for all entries
            in the entry set.
        MP_API_KEY: API key for Materials Project. Note: if not provided, MPRester will
            automatically look for an environment variable.
    """

    name: str = "get_and_process_entries"
    entry_db_name: str = "entries_db"
    temperature: int = 300
    include_nist_data: bool = True
    include_barin_data: bool = False
    include_freed_data: bool = False
    e_above_hull: float = 0.0
    include_polymorphs: bool = False
    formulas_to_include: list = field(default_factory=list)
    calculate_e_above_hulls: bool = True
    MP_API_KEY: Optional[str] = None
    property_data: Optional[List[str]] = None

    @job(entries="entries", output_schema=EntrySetDocument)
    def make(self, chemsys):
        entry_db = SETTINGS.JOB_STORE.additional_stores.get(self.entry_db_name)

        if entry_db:
            property_data = self.property_data
            if property_data is None:
                property_data = ["theoretical"]
            elif "theoretical" not in property_data:
                property_data.append("theoretical")

            entries = get_all_entries_in_chemsys(
                entry_db,
                chemsys,
                inc_structure=True,
                compatible_only=True,
                property_data=property_data,
                use_premade_entries=False,
            )
        else:
            from mp_api.client import MPRester

            if self.MP_API_KEY:
                with MPRester(self.MP_API_KEY) as mpr:
                    entries = mpr.get_entries_in_chemsys(elements=chemsys)
            else:
                with MPRester() as mpr:  # let MPRester look for env variable
                    entries = mpr.get_entries_in_chemsys(elements=chemsys)

        entries = process_entries(
            entries,
            temperature=self.temperature,
            include_nist_data=self.include_nist_data,
            include_barin_data=self.include_barin_data,
            include_freed_data=self.include_freed_data,
            e_above_hull=self.e_above_hull,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
            calculate_e_above_hulls=self.calculate_e_above_hulls,
        )

        doc = EntrySetDocument(
            entries=entries,
            e_above_hull=self.e_above_hull,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
        )
        doc.task_label = self.name

        return doc


@dataclass
class ReactionEnumerationMaker(Maker):
    """
    Maker to create job for enumerating reactions from a set of entries.
    """

    name: str = "enumerate_reactions"

    @job(rxns="rxns", output_schema=EnumeratorTaskDocument)
    def make(self, enumerators, entries):
        data = {}

        logger.info("Running enumerators...")
        data["rxns"] = run_enumerators(enumerators, entries)
        data.update(self._get_metadata(enumerators, entries))

        logger.info("Building task document...")
        enumerator_task = EnumeratorTaskDocument(**data)
        enumerator_task.task_label = self.name
        return enumerator_task

    def _get_metadata(self, enumerators, entries):
        chemsys = "-".join(entries.chemsys)
        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }

        added_elements = None
        added_chemsys = None

        if targets:
            added_elements, added_chemsys = get_added_elem_data(entries, targets)

        metadata = {
            "elements": [Element(e) for e in chemsys.split("-")],
            "chemsys": chemsys,
            "enumerators": [e.as_dict() for e in enumerators],
            "targets": list(sorted(targets)),
            "added_elements": added_elements,
            "added_chemsys": added_chemsys,
        }
        return metadata


@dataclass
class CalculateSelectivitiesMaker(Maker):
    """Maker to create job for calculating selectivities of a set of reactions and
    target formula."""

    name: str = "calculate_selectivities"
    open_elem: Optional[Element] = None
    chempot: Optional[float] = 0.0
    calculate_selectivities: bool = True
    calculate_chempot_distances: bool = True
    temp: float = 300.0
    chunk_size: Optional[int] = None
    batch_size: Optional[int] = None
    cpd_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = (
            Composition(str(self.open_elem)).reduced_formula if self.open_elem else None
        )

    @job(rxns="rxns", output_schema=SelectivitiesTaskDocument)
    def make(self, rxn_sets, entries, target_formula):
        initialize_ray()

        target_formula = Composition(target_formula).reduced_formula
        added_elements, added_chemsys = get_added_elem_data(entries, [target_formula])

        logger.info("Loading reactions..")
        all_rxns = rxn_sets[0]
        for rxn_set in rxn_sets[1:]:
            all_rxns = all_rxns.add_rxn_set(rxn_set)

        size = len(all_rxns)  # need to get size before storing in ray

        logger.info("Identifying target reactions...")

        target_rxns = ReactionSet.from_rxns(
            list(all_rxns.get_rxns_by_product(target_formula))
        )
        logger.info(
            f"Identified {len(target_rxns)} target reactions out of"
            f" {size} total reactions."
        )
        logger.info("Placing reactions in ray object store...")

        all_rxns = ray.put(all_rxns)
        logger.info("Beginning selectivity calculations...")

        decorated_rxns = target_rxns

        if self.calculate_selectivities:
            decorated_rxns = self._get_selectivity_decorated_rxns(
                target_rxns, all_rxns, size
            )

        logger.info("Calculating chemical potential distances...")

        if self.calculate_chempot_distances:
            decorated_rxns = self._get_chempot_decorated_rxns(decorated_rxns, entries)

        logger.info("Saving decorated reactions.")
        results = ReactionSet.from_rxns(decorated_rxns, entries=entries)

        data = {
            "rxns": results,
            "target_formula": target_formula,
            "open_elem": self.open_elem,
            "chempot": self.chempot,
            "added_elements": added_elements,
            "added_chemsys": added_chemsys,
            "calculate_selectivities": self.calculate_selectivities,
            "calculate_chempot_distances": self.calculate_chempot_distances,
            "temp": self.temp,
            "batch_size": self.batch_size,
            "cpd_kwargs": self.cpd_kwargs,
        }

        doc = SelectivitiesTaskDocument(**data)
        doc.task_label = self.name
        return doc

    def _get_selectivity_decorated_rxns(self, target_rxns, all_rxns, size):
        memory_per_rxn = 800  # estimate of 800 bytes memory per rxn

        memory_size = int(ray.cluster_resources()["memory"])
        logger.info(f"Available memory: {memory_size}")

        num_cpus = int(ray.cluster_resources()["CPU"]) - 1
        logger.info(f"Available CPUs - 1: {num_cpus}")

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = num_cpus
            if memory_size < (size * memory_per_rxn * num_cpus):
                batch_size = memory_size // (size * memory_per_rxn)
                logger.info(
                    f"Memory size too small for {num_cpus} batches. "
                    f"Using batch size of {batch_size}."
                )

        chunk_size = self.chunk_size or (len(target_rxns) // batch_size) + 1

        logger.info(f"Using batch size of {batch_size} and chunk size of {chunk_size}")

        rxn_chunk_refs = []
        results = []

        with tqdm(total=len(target_rxns) // chunk_size + 1) as pbar:
            for chunk in grouper(
                target_rxns,
                chunk_size,
                fillvalue=None,
            ):
                chunk = [c for c in chunk if c is not None]
                if len(rxn_chunk_refs) > batch_size:
                    num_ready = len(rxn_chunk_refs) - batch_size
                    newly_completed, rxn_chunk_refs = ray.wait(
                        rxn_chunk_refs, num_returns=num_ready
                    )
                    for completed_ref in newly_completed:
                        results.append(ray.get(completed_ref))
                        pbar.update(1)

                reactant_formulas = [
                    c.reduced_formula for rxn in chunk for c in rxn.reactants
                ]
                if self.open_formula:
                    reactant_formulas.append(self.open_formula)

                task_memory = memory_per_rxn * (size)

                rxn_chunk_refs.append(
                    _get_selectivity_decorated_rxns_by_chunk.options(
                        memory=task_memory
                    ).remote(chunk, all_rxns, self.open_formula, self.temp)
                )

            newly_completed, rxn_chunk_refs = ray.wait(
                rxn_chunk_refs, num_returns=len(rxn_chunk_refs)
            )
            for completed_ref in newly_completed:
                results.append(ray.get(completed_ref))
                pbar.update(1)

        decorated_rxns = [rxn for r_set in results for rxn in r_set]

        return ReactionSet.from_rxns(decorated_rxns)

    def _get_chempot_decorated_rxns(self, target_rxns, entries):
        initialize_ray()

        batch_size = self.batch_size or int(ray.cluster_resources()["CPU"] - 1)

        chunk_size = self.chunk_size or (len(target_rxns) // batch_size) + 1
        rxn_chunk_refs = []
        results = []

        open_elem = target_rxns.open_elem
        chempot = target_rxns.chempot

        entries = ray.put(entries)
        cpd_kwargs = ray.put(self.cpd_kwargs)

        with tqdm(total=len(target_rxns) // chunk_size + 1) as pbar:
            for chunk in grouper(
                sorted(target_rxns, key=lambda r: r.chemical_system),
                chunk_size,
                fillvalue=None,
            ):
                chunk = [c for c in chunk if c is not None]
                if len(rxn_chunk_refs) > batch_size:
                    num_ready = len(rxn_chunk_refs) - batch_size
                    newly_completed, rxn_chunk_refs = ray.wait(
                        rxn_chunk_refs, num_returns=num_ready
                    )
                    for completed_ref in newly_completed:
                        results.append(ray.get(completed_ref))
                        pbar.update(1)

                rxn_chunk_refs.append(
                    _get_chempot_decorated_rxns_by_chunk.remote(
                        chunk, entries, cpd_kwargs, open_elem, chempot
                    )
                )

            newly_completed, rxn_chunk_refs = ray.wait(
                rxn_chunk_refs, num_returns=len(rxn_chunk_refs)
            )
            for completed_ref in newly_completed:
                results.append(ray.get(completed_ref))
                pbar.update(1)

        decorated_rxns = [rxn for r_set in results for rxn in r_set]

        return ReactionSet.from_rxns(decorated_rxns)


@dataclass
class NetworkMaker(Maker):
    """
    Maker for generating reaction networks and performing pathfinding from a set of
    previously enumerated reactions.
    """

    name: str = "build_and_analyze_network"
    cost_function: CostFunction = field(default_factory=Softplus)
    precursors: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    calculate_pathways: Optional[int] = 10
    open_elem: Optional[Element] = None
    chempot: float = 0.0

    @job(network="network", output_schema=NetworkTaskDocument)
    def make(
        self,
        rxn_sets: Iterable[ReactionSet],
    ):
        all_rxns = ReactionSet.from_rxns(
            [rxn for rxn_set in rxn_sets for rxn in rxn_set.get_rxns()],
            open_elem=self.open_elem,
            chempot=self.chempot,
        )
        all_rxns = all_rxns.filter_duplicates()

        rn = ReactionNetwork(all_rxns, cost_function=self.cost_function)
        rn.build()

        paths = None
        if self.precursors:
            rn.set_precursors(self.precursors)
        if self.targets:
            rn.set_target(self.targets[0])
        if self.calculate_pathways and self.targets:
            paths = rn.find_pathways(self.targets, k=self.calculate_pathways)

        data = {
            "network": rn,
            "paths": paths,
            "k": self.calculate_pathways,
            "precursors": self.precursors,
            "targets": self.targets,
        }
        doc = NetworkTaskDocument(**data)
        doc.task_label = self.name
        return doc


@dataclass
class PathwaySolverMaker(Maker):
    """
    Maker for solving balanced reaction pathways from a set of (unbalanced) pathways
    returned as part of reaction network analysis.
    """

    name: str = "solve pathways"
    cost_function: CostFunction = field(default_factory=Softplus)
    precursors: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    open_elem: Optional[Element] = None
    chempot: Optional[float] = None
    max_num_combos: int = 4
    find_intermediate_rxns: bool = True
    intermediate_rxn_energy_cutoff: float = 0.0
    use_basic_enumerator: bool = True
    use_minimize_enumerator: bool = False
    filter_interdependent: bool = True

    def __post_init__(self):
        net_rxn = BasicReaction.balance(
            [Composition(r) for r in self.precursors],
            [Composition(p) for p in self.targets],
        )
        if not net_rxn.balanced:
            raise ValueError(
                "Can not balance pathways with specified precursors/targets. Please"
                " make sure a balanced net reaction can be written from the provided"
                " reactant and product formulas!"
            )
        self.net_rxn = net_rxn

    @job(paths="paths", output_schema=PathwaySolverTaskDocument)
    def make(self, pathways, entries):
        chempots = None
        if self.open_elem:
            chempots = {Element(self.open_elem): self.chempot}

        ps = PathwaySolver(
            pathways=pathways,
            entries=entries,
            cost_function=self.cost_function,
            open_elem=self.open_elem,
            chempot=self.chempot,
        )

        net_rxn = get_computed_rxn(self.net_rxn, entries, chempots)

        balanced_paths = ps.solve(
            net_rxn=net_rxn,
            max_num_combos=self.max_num_combos,
            find_intermediate_rxns=self.find_intermediate_rxns,
            intermediate_rxn_energy_cutoff=self.intermediate_rxn_energy_cutoff,
            use_basic_enumerator=self.use_basic_enumerator,
            use_minimize_enumerator=self.use_minimize_enumerator,
            filter_interdependent=self.filter_interdependent,
        )
        data = {
            "solver": ps,
            "balanced_paths": balanced_paths,
            "precursors": self.precursors,
            "targets": self.targets,
            "net_rxn": net_rxn,
            "max_num_combos": self.max_num_combos,
            "find_intermediate_rxns": self.find_intermediate_rxns,
            "intermediate_rxn_energy_cutoff": self.intermediate_rxn_energy_cutoff,
            "use_basic_enumerator": self.use_basic_enumerator,
            "use_minimize_enumerator": self.use_minimize_enumerator,
            "filter_interdependent": self.filter_interdependent,
        }

        doc = PathwaySolverTaskDocument(**data)
        doc.task_label = self.name

        return doc


@ray.remote
def _get_selectivity_decorated_rxns_by_chunk(rxn_chunk, all_rxns, open_formula, temp):
    decorated_rxns = []

    for rxn in rxn_chunk:
        if not rxn:
            continue

        reactant_formulas = [r.reduced_formula for r in rxn.reactants]
        reactants_with_open = reactant_formulas.copy()

        if open_formula:
            reactants_with_open.append(open_formula)

        competing_rxns = all_rxns.get_rxns_by_reactants(reactants_with_open)

        if len(reactant_formulas) >= 3:
            if open_formula:
                reactant_formulas = list(set(reactant_formulas) - {open_formula})
            else:
                raise ValueError("Can only have 2 precursors, excluding open element!")

        decorated_rxns.append(
            _get_selectivity_decorated_rxn(rxn, competing_rxns, reactant_formulas, temp)
        )

    return decorated_rxns


def _get_selectivity_decorated_rxn(rxn, competing_rxns, precursors_list, temp):
    """ """
    if len(precursors_list) == 1:
        other_energies = np.array(
            [r.energy_per_atom for r in competing_rxns if r != rxn]
        )
        primary_selectivity = InterfaceReactionHull._primary_selectivity_from_energies(  # pylint: disable=protected-access, line-too-long # noqa: E501
            rxn.energy_per_atom, other_energies, temp=temp
        )
        energy_diffs = rxn.energy_per_atom - np.append(
            other_energies, 0.0
        )  # consider identity reaction as well

        secondary_rxn_energies = energy_diffs[energy_diffs > 0]
        secondary_selectivity = (
            secondary_rxn_energies.max() if secondary_rxn_energies.any() else 0.0
        )
        rxn.data["primary_selectivity"] = round(primary_selectivity, 4)
        rxn.data["secondary_selectivity"] = round(secondary_selectivity, 4)
        rxn.data["secondary_selectivity_max"] = round(secondary_selectivity, 4)
        rxn.data["secondary_selectivity_area"] = round(secondary_selectivity, 4)
        decorated_rxn = rxn
    else:
        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            list(competing_rxns),
        )

        calc_1 = PrimarySelectivityCalculator(irh=irh, temp=temp)
        calc_2 = SecondarySelectivityCalculator(irh=irh)
        calc_3 = SecondarySelectivityMaxCalculator(irh=irh)
        calc_4 = SecondarySelectivityAreaCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)
        decorated_rxn = calc_3.decorate(decorated_rxn)
        decorated_rxn = calc_4.decorate(decorated_rxn)

    return decorated_rxn


@ray.remote
def _get_chempot_decorated_rxns_by_chunk(
    rxn_chunk, entries, cpd_kwargs, open_elem, chempot
):
    cpd_calc_dict = {}
    new_rxns = []

    if open_elem:
        open_elem_set = {open_elem}

    for rxn in sorted(rxn_chunk, key=lambda rxn: len(rxn.elements), reverse=True):
        chemsys = rxn.chemical_system
        elems = chemsys.split("-")

        for c, cpd_calc in cpd_calc_dict.items():
            if set(c.split("-")).issuperset(elems):
                break
        else:
            if open_elem:
                filtered_entries = entries.get_subset_in_chemsys(
                    elems + [str(open_elem)]
                )
                filtered_entries = [
                    e.to_grand_entry({Element(open_elem): chempot})
                    for e in filtered_entries
                    if set(e.composition.elements) != open_elem_set
                ]
            else:
                filtered_entries = entries.get_subset_in_chemsys(elems)

            cpd_calc = ChempotDistanceCalculator.from_entries(
                filtered_entries, **cpd_kwargs
            )
            cpd_calc_dict[chemsys] = cpd_calc

        new_rxn = cpd_calc.decorate(rxn)
        new_rxns.append(new_rxn)

    results = ReactionSet.from_rxns(new_rxns, entries=entries)
    return results
