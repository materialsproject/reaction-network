"""Core jobs for reaction network creation and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import ray
from jobflow import SETTINGS, Maker, job
from pymatgen.core.composition import Element

from rxn_network.core import Composition
from rxn_network.costs.calculators import (
    ChempotDistanceCalculator,
    PrimaryCompetitionCalculator,
    SecondaryCompetitionCalculator,
)
from rxn_network.costs.functions import Softplus
from rxn_network.entries.utils import get_all_entries_in_chemsys_from_entry_db, process_entries
from rxn_network.enumerators.utils import get_computed_rxn, run_enumerators
from rxn_network.jobs.schema import (
    EntrySetDocument,
    EnumeratorTaskDocument,
    NetworkTaskDocument,
    PathwaySolverTaskDocument,
)
from rxn_network.jobs.utils import get_added_elem_data
from rxn_network.network.network import ReactionNetwork
from rxn_network.pathways.solver import PathwaySolver
from rxn_network.reactions.basic import BasicReaction
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rxn_network.costs.base import CostFunction
    from rxn_network.entries.entry_set import GibbsEntrySet
    from rxn_network.enumerators.base import Enumerator
    from rxn_network.pathways.pathway_set import PathwaySet

logger = get_logger(__name__)


@dataclass
class GetEntrySetMaker(Maker):
    """Maker to create job for acquiring and processing entries to be used in reaction
    enumeration or network building. See GibbsEntrySet for more information about how
    these are constructed.

    Args:
        name: Name of the job.
        temperature: Temperature [K] for determining Gibbs Free Energy of
            formation, dGf(T).
        e_above_hull: Energy above hull (eV/atom) for thermodynamic stability threshold;
            i.e., include all entries with energies below this value.
        filter_at_temperature: Temperature (in Kelvin) at which entries are filtered for
            thermodynamic stability (e.g., room temperature). Generally, this often
            differs from the synthesis temperature.
        include_nist_data: Whether to include NIST-JANAF data in the entry set.
            Defaults to True.
        include_freed_data: Whether to include FREED data in the entry set. Defaults
            to False. WARNING: This dataset has not been thoroughly tested. Use at
            your own risk!
        formulas_to_include: An iterable of compositional formulas to ensure are
            included in the processed dataset. Sometimes, entries are filtered out that
            one would like to include, or entries don't exist for those compositions.
        calculate_e_above_hulls: Whether to calculate e_above_hull and store as an
            attribute in the data dictionary for each entry. Defaults to True.
        ignore_nist_solids: Whether to ignore NIST data for solids with high melting
            points (Tm >= 1500 ºC). Defaults to True.
        custom_entries: An optional list of user-created entries that will be added to
            the final entry set.
        MP_API_KEY: The Materials Project API key to use for querying MP using mp-api.
            If not provided, this will default to whatever is stored through
            configuration settings on the user's machine.
        entry_db_name (for internal use only): if not None, then this method will use an
            internal materials databse using the get_all_entries_in_chemsys() method. If
            you wish to use this approach, you may need to reconfigure that method.
        property_data: a list of attributes to ensure are included in each entry's data,
            if available. This currently only applies to those using a custom entry DB
            (see above).
    """

    name: str = "get_and_process_entries"
    temperature: int = 300
    e_above_hull: float = 0.0
    filter_at_temperature: int | None = None
    include_nist_data: bool = True
    include_freed_data: bool = False
    include_polymorphs: bool = False
    formulas_to_include: list[str] = field(default_factory=list)
    calculate_e_above_hulls: bool = True
    ignore_nist_solids: bool = True
    custom_entries: list = field(default_factory=list)
    MP_API_KEY: str | None = None
    entry_db_name: str | None = None
    property_data: list[str] | None = None

    @job(entries="entries", output_schema=EntrySetDocument)
    def make(self, chemsys: str):
        """Returns a job that acquires a GibbsEntrySet for the desired chemical system.

        NOTE: This job stores the entry set in an additional store called
        "entries". This needs to be configured through a user's jobflow.yaml file. See
        "additional_stores".

        Args:
            chemsys: The chemical system of the entry set to be acquired.
        """
        entry_db = SETTINGS.JOB_STORE.additional_stores.get(self.entry_db_name)  # pylint: disable=no-member

        if entry_db:
            logger.info(f"Using user-specified Entry DB: {self.entry_db_name}")
            property_data = self.property_data
            if property_data is None:
                property_data = ["theoretical"]
            elif "theoretical" not in property_data:
                property_data.append("theoretical")

            entries = get_all_entries_in_chemsys_from_entry_db(
                entry_db,
                chemsys,
            )
        else:
            try:
                from mp_api.client import MPRester
            except ImportError as err:
                raise ImportError("You may need to install the Materials Project API: pip install -U mp-api") from err

            kwargs = {}
            if self.MP_API_KEY:
                kwargs["api_key"] = self.MP_API_KEY

            elems = {Element(i) for i in chemsys.split("-")}

            if len(elems) <= 25:
                with MPRester(**kwargs) as mpr:
                    entries = mpr.get_entries_in_chemsys(
                        elements=chemsys,
                        additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                    )
            else:  # this approach is faster for big systems
                other_elems = self._get_exclude_elems(elems)

                with MPRester(**kwargs) as mpr:
                    docs = mpr.summary.search(exclude_elements=other_elems, all_fields=False, deprecated=False)
                mpids = [d.material_id for d in docs]

                with MPRester(**kwargs) as mpr:
                    entries = mpr.get_entries(mpids, additional_criteria={"thermo_types": ["GGA_GGA+U"]})

        if self.custom_entries:
            entries.extend(self.custom_entries)

        entries = process_entries(
            entries,
            temperature=self.temperature,
            e_above_hull=self.e_above_hull,
            include_nist_data=self.include_nist_data,
            include_freed_data=self.include_freed_data,
            filter_at_temperature=self.filter_at_temperature,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
            calculate_e_above_hulls=self.calculate_e_above_hulls,
            ignore_nist_solids=self.ignore_nist_solids,
        )

        doc = EntrySetDocument(
            entries=entries,
            e_above_hull=self.e_above_hull,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
        )
        doc.task_label = self.name

        return doc

    @staticmethod
    def _get_exclude_elems(elems):
        """Get the inverse element selection. Helpful for faster querying of very large
        chemical systems).
        """
        exclude_elems = []
        for e in Element:
            if e in elems:
                continue
            exclude_elems.append(str(e))

        return exclude_elems


@dataclass
class ReactionEnumerationMaker(Maker):
    """Maker to create a job for enumerating reactions from a set of entries. This is
    effectively a wrapper to the run_enumerators() function (see enumerators.utils) with
    a defined output document.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.

    Args:
        name: The name of the job. This is automatically created if not provided.
    """

    name: str = "enumerate_reactions"

    @job(rxns="rxns", output_schema=EnumeratorTaskDocument)
    def make(self, enumerators: Iterable[Enumerator], entries: GibbsEntrySet):
        """Returns a job that enumerates reactions from a set of entries.

        NOTE: This job stores the reaction set in an additional store called
        "rxns". This needs to be configured through a user's jobflow.yaml file. See
        "additional_stores".

        Args:
            enumerators: An iterable of enumerators to perform enumeration (e.g.,
                [BasicEnumerator, BasicOpenEnumerator])
            entries: An entry set provided to each of the enumerators
        """
        data = {}

        logger.info("Running enumerators...")
        data["rxns"] = run_enumerators(enumerators, entries)
        data.update(self._get_metadata(enumerators, entries))

        logger.info("Building task document...")
        enumerator_task = EnumeratorTaskDocument(**data)
        enumerator_task.task_label = self.name
        return enumerator_task

    def _get_metadata(self, enumerators: Iterable[Enumerator], entries: GibbsEntrySet):
        """Acquires common metadata provided to output document."""
        chemsys = "-".join(entries.chemsys)
        targets = {target for enumerator in enumerators for target in enumerator.targets}

        added_elements = None
        added_chemsys = None

        if targets:
            added_elements, added_chemsys = get_added_elem_data(entries, targets)

        return {
            "elements": [Element(e) for e in chemsys.split("-")],
            "chemsys": chemsys,
            "enumerators": [e.as_dict() for e in enumerators],
            "targets": sorted(targets),
            "added_elements": added_elements,
            "added_chemsys": added_chemsys,
        }


@dataclass
class NetworkMaker(Maker):
    """Maker for generating reaction networks and performing pathfinding from a set of
    previously enumerated reactions. This is a component of the NetworkFlowMaker.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.

    Args:
        name: The name of the job. Automatically selected if not provided.
        cost_function: The cost function used to calculate the edge weights in the
            network.
        precursors: An optional list of precursor formulas used in call to
            set_precursors().
        targets: An optional list of target formulas used in calls to set_target() as
            part of pathfinding.
        calculate_pathways: If integer is provided, basic pathfinding is performed using
            the find_pathways() function up to the provided number of paths (k).
            Defaults to k=10. Provide a value of None if no pathfinding is to be
            performed.
        open_elem: An optional open element used to re-initialize the total reaction
            set.
        chempot: The chemical potential of the open element, if any. Defaults to 0.
    """

    name: str = "build_and_analyze_network"
    cost_function: CostFunction = field(default_factory=Softplus)
    precursors: list[str] | None = None
    targets: list[str] | None = None
    calculate_pathways: int | None = 10
    open_elem: Element | None = None
    chempot: float = 0.0

    @job(network="network", output_schema=NetworkTaskDocument)
    def make(
        self,
        rxn_sets: Iterable[ReactionSet],
    ):
        """Returns a job that creates a reaction network from an iterable of reaction sets
        (coming from reaction enumeration).

        NOTE: This job stores the network object in an additional store called
        "network". This needs to be configured through a user's jobflow.yaml file. See
        "additional_stores".

        Args:
            rxn_sets: An iterable of reaction sets produced by reaction enumeration.
                These are combined, re-initialized according to provided environmental
                conditions, and then used to build the network.

        Returns:
            A job that builds and stores the reaction network for enumerated reactions.
            Also performs basic pathfinding (but no balancing; see PathwaySolverMaker).

        """
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
    """Maker for solving balanced reaction pathways from a set of (unbalanced) pathways
    returned as part of reaction network analysis.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.

    Args:
        name: The name of the job. Automatically selected if not provided.
        cost_function: The cost function used to calculate the weights of any new edges
            identified during solving (e.g., for intermediate reactions).
        precursors: The precursor formulas for the overall net reaction. These must be
            provided and result in a valid net reaction (together with the targets).
        targets: The target formulas for the overall net reaction. These must be
            provided and result in a valid net reaction (together with the precursors).
        open_elem: An optional open element used by PathwaySolver.
        chempot: The chemical potential of the open element, if any. Defaults to 0.
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
    """

    name: str = "solve pathways"
    cost_function: CostFunction = field(default_factory=Softplus)
    precursors: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    open_elem: Element | None = None
    chempot: float = 0.0
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
    def make(
        self,
        pathways: PathwaySet,
        entries: GibbsEntrySet,
    ):
        """Returns a job that produces balanced pathways from raw pathways coming out of
        network pathfinding.

        NOTE: This job stores the network object in an additional store called
        "paths". This needs to be configured through a user's jobflow.yaml file. See
        "additional_stores".

        Args:
            pathways: A set of pathways (as returned by find_pathways()).
            entries: The entry set to use in path solving.

        """
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


def _get_competition_decorated_rxn(rxn, competing_rxns, precursors_list):
    """Calculates the primary and secondary competition for a reaction and stores them
    within the data dict.
    """
    if len(precursors_list) == 1:
        energy = rxn.energy_per_atom
        other_energies = np.array([r.energy_per_atom for r in competing_rxns if r != rxn])
        if not np.isclose(other_energies, 0.0).any():
            other_energies = np.append(other_energies, 0.0)  # need identity reaction

        primary_competition = energy - other_energies.min()

        energy_diffs = rxn.energy_per_atom - other_energies
        secondary_rxn_energies = energy_diffs[energy_diffs > 0]
        secondary_competition = secondary_rxn_energies.max() if secondary_rxn_energies.any() else 0.0
        rxn.data["primary_competition"] = round(primary_competition, 4)
        rxn.data["secondary_competition"] = round(secondary_competition, 4)

        decorated_rxn = rxn
    else:
        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            list(competing_rxns),
        )

        calc_1 = PrimaryCompetitionCalculator(irh=irh)
        calc_2 = SecondaryCompetitionCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)

    return decorated_rxn


@ray.remote
def _get_competition_decorated_rxns_by_chunk(rxn_chunk, all_rxns, open_formula):
    """Performs competition score calculations within a chunk of reactions."""
    decorated_rxns = []
    all_rxns = ReactionSet.from_dict(all_rxns)  # stored in Ray as a dict (should be memory efficient)

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

        decorated_rxns.append(_get_competition_decorated_rxn(rxn, competing_rxns, reactant_formulas))

    return decorated_rxns


@ray.remote
def _get_chempot_decorated_rxns_by_chunk(rxn_chunk, entries, cpd_kwargs, open_elem, chempot):
    """Calculates total chemical potential distance for a chunk of reactions."""
    cpd_calc_dict = {}
    new_rxns = []

    if open_elem:
        open_elem_set = {open_elem}

    rxn_chunk = [r for r in rxn_chunk if r is not None]

    for rxn in sorted(rxn_chunk, key=lambda rxn: len(rxn.elements), reverse=True):
        chemsys = rxn.chemical_system
        elems = chemsys.split("-")

        for c, cpd_calculator in cpd_calc_dict.items():
            if set(c.split("-")).issuperset(elems):
                cpd_calc = cpd_calculator
                break
        else:
            if open_elem:
                filtered_entries = entries.get_subset_in_chemsys([*elems, str(open_elem)])
                filtered_entries = [
                    e.to_grand_entry({Element(open_elem): chempot})
                    for e in filtered_entries
                    if set(e.composition.elements) != open_elem_set
                ]
            else:
                filtered_entries = entries.get_subset_in_chemsys(elems)

            cpd_calc = ChempotDistanceCalculator.from_entries(filtered_entries, **cpd_kwargs)
            cpd_calc_dict[chemsys] = cpd_calc

        new_rxn = cpd_calc.decorate(rxn)
        new_rxns.append(new_rxn)

    return ReactionSet.from_rxns(new_rxns, entries=entries)
