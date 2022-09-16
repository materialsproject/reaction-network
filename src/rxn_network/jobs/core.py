"""Core jobs for reaction-network creation and analysis."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Iterable
from pathlib import Path

import numpy as np
import ray
from jobflow import SETTINGS, Maker, job
from pymatgen.core.composition import Element
from tqdm import tqdm

from rxn_network.core.composition import Composition
from rxn_network.costs.calculators import (
    ChempotDistanceCalculator,
    PrimarySelectivityCalculator,
    SecondarySelectivityCalculator,
)
from rxn_network.core.cost_function import CostFunction
from rxn_network.costs.softplus import Softplus
from rxn_network.entries.utils import get_all_entries_in_chemsys, process_entries
from rxn_network.jobs.schema import (
    EntrySetDocument,
    EnumeratorTaskDocument,
    NetworkTaskDocument,
    SelectivitiesTaskDocument,
    PathwaySolverTaskDocument,
)
from rxn_network.jobs.utils import (
    get_added_elem_data,
    run_enumerators,
)
from rxn_network.network.network import ReactionNetwork
from rxn_network.reactions.basic import BasicReaction
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.utils import grouper
from rxn_network.utils.ray import initialize_ray

logger = logging.getLogger(__name__)


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
        calculate_e_above_hulls: Whether to calculate e_above_hulls for all entries in the entry set.
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

        property_data = self.property_data
        if property_data is None:
            property_data = ["theoretical"]
        elif "theoretical" not in property_data:
            property_data.append("theoretical")

        if entry_db:
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

            with MPRester(self.MP_API_KEY) as mpr:
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
        data["rxns"] = run_enumerators(enumerators, entries)
        data.update(self._get_metadata(enumerators, entries))

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
    chunk_size: int = 20
    batch_size: Optional[int] = None
    cpd_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = (
            Composition(str(self.open_elem)).reduced_formula if self.open_elem else None
        )

    @job(rxns="rxns", output_schema=SelectivitiesTaskDocument)
    def make(self, rxn_sets, entries, target_formula):
        target_formula = Composition(target_formula).reduced_formula
        added_elements, added_chemsys = get_added_elem_data(entries, [target_formula])

        logger.info("Identifying target reactions...")
        all_rxns = ReactionSet.from_rxns(
            [rxn for rxn_set in rxn_sets for rxn in rxn_set.get_rxns()],
            entries=entries,
            open_elem=self.open_elem,
            chempot=self.chempot,
        )
        all_rxns = all_rxns.filter_duplicates()

        target_rxns = []
        for rxn in all_rxns:
            product_formulas = [p.reduced_formula for p in rxn.products]
            if target_formula in product_formulas:
                target_rxns.append(rxn)

        target_rxns = ReactionSet.from_rxns(target_rxns)

        logger.info(
            f"Identified {len(target_rxns)} target reactions out of"
            f" {len(all_rxns)} total reactions."
        )

        decorated_rxns = target_rxns

        if self.calculate_selectivities:
            decorated_rxns = self._get_selectivity_decorated_rxns(target_rxns, all_rxns)

        logger.info("Saving decorated reactions.")

        if self.calculate_chempot_distances:
            decorated_rxns = self._get_chempot_decorated_rxns(decorated_rxns, entries)

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

    def _get_selectivity_decorated_rxns(self, target_rxns, all_rxns):
        initialize_ray()
        all_rxns = ray.put(all_rxns)

        batch_size = self.batch_size or ray.cluster_resources()["CPU"]

        logger.info("Calculating selectivites...")

        rxn_chunk_refs = []
        results = []
        with tqdm(total=len(target_rxns) // self.chunk_size + 1) as pbar:
            for chunk in grouper(
                target_rxns.get_rxns(), self.chunk_size, fillvalue=None
            ):
                if len(rxn_chunk_refs) > batch_size:
                    num_ready = len(rxn_chunk_refs) - batch_size
                    newly_completed, rxn_chunk_refs = ray.wait(
                        rxn_chunk_refs, num_returns=num_ready
                    )
                    for completed_ref in newly_completed:
                        results.append(ray.get(completed_ref))
                        pbar.update(1)

                rxn_chunk_refs.append(
                    _get_selectivity_decorated_rxns_by_chunk.remote(
                        chunk, all_rxns, self.open_formula, self.temp
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

    def _get_chempot_decorated_rxns(self, rxns, entries):
        cpd_calc_dict = {}
        new_rxns = []

        open_elem = rxns.open_elem
        if open_elem:
            open_elem_set = {open_elem}
        chempot = rxns.chempot

        for rxn in sorted(rxns, key=lambda rxn: len(rxn.elements), reverse=True):
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
                    filtered_entries, **self.cpd_kwargs
                )
                cpd_calc_dict[chemsys] = cpd_calc

            new_rxns.append(cpd_calc.decorate(rxn))

        results = ReactionSet.from_rxns(new_rxns, entries=entries)
        return results


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
    graph_fn: Optional[str] = None

    @job(output_schema=NetworkTaskDocument)
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

        if self.precursors:
            rn.set_precursors(self.precursors)
        if self.targets:
            rn.set_target(self.targets[0])
        if self.calculate_pathways and self.targets:
            paths = rn.find_pathways(self.targets, k=self.calculate_pathways)

        graph_fn = self.graph_fn or "network.gt.gz"
        graph_fn = str(Path(graph_fn).absolute())
        rn.graph.save(graph_fn)

        data = {
            "network": rn,
            "graph_fn": graph_fn,
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
    precursors: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
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
        net_rxn = get_computed_rxn(self.net_rxn, entries, chempots)

        ps = PathwaySolver(
            pathways=pathways,
            entries=entries,
            cost_function=self.cost_function,
            open_elem=self.open_elem,
            chempot=self.chempot,
        )
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

        precursors = [r.reduced_formula for r in rxn.reactants]
        competing_rxns = list(all_rxns.get_rxns_by_reactants(precursors))

        if open_formula:
            open_formula = Composition(open_formula).reduced_formula
            competing_rxns.extend(
                all_rxns.get_rxns_by_reactants(precursors + [open_formula])
            )

        if len(precursors) >= 3:
            precursors = list(set(precursors) - {open_formula})

        decorated_rxns.append(
            _get_selectivity_decorated_rxn(rxn, competing_rxns, precursors, temp)
        )

    return decorated_rxns


def _get_selectivity_decorated_rxn(rxn, competing_rxns, precursors_list, temp):
    """ """
    if len(precursors_list) == 1:
        other_energies = np.array(
            [r.energy_per_atom for r in competing_rxns if r != rxn]
        )
        primary_selectivity = InterfaceReactionHull._primary_selectivity_from_energies(  # pylint: disable=protected-access
            rxn.energy_per_atom, other_energies, temp=temp
        )
        energy_diffs = rxn.energy_per_atom - np.append(
            other_energies, 0.0
        )  # consider identity reaction as well

        secondary_rxn_energies = energy_diffs[energy_diffs > 0]
        secondary_selectivity = (
            secondary_rxn_energies.max() if secondary_rxn_energies.any() else 0.0
        )
        rxn.data["primary_selectivity"] = primary_selectivity
        rxn.data["secondary_selectivity"] = secondary_selectivity
        decorated_rxn = rxn
    else:
        if rxn not in competing_rxns:
            competing_rxns.append(rxn)

        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            competing_rxns,
        )

        calc_1 = PrimarySelectivityCalculator(irh=irh, temp=temp)
        calc_2 = SecondarySelectivityCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)

    return decorated_rxn
