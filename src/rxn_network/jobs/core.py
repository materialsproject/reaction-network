"""Core jobs for reaction-network creation and analysis."""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from jobflow import SETTINGS, Maker, job
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.costs.softplus import Softplus
from rxn_network.entries.utils import get_all_entries_in_chemsys, process_entries
from rxn_network.enumerators.utils import get_computed_rxn
from rxn_network.jobs.schema import (
    EntrySetDocument,
    EnumeratorTaskDocument,
    NetworkTaskDocument,
    PathwaySolverTaskDocument,
)
from rxn_network.jobs.utils import get_added_elem_data, run_enumerators
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.basic import BasicReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger

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
