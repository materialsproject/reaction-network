import logging
from dataclasses import dataclass, field
from typing import Collection, List, Optional

from jobflow import Flow, Maker, job
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.core.enumerator import Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.jobs.core import (
    CalculateSelectivitiesMaker,
    GetEntrySetMaker,
    NetworkMaker,
    ReactionEnumerationMaker,
    PathwaySolverMaker,
)
from rxn_network.jobs.schema import EnumeratorTaskDocument, NetworkTaskDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrosynthesisFlowMaker(Maker):
    name: str = "identify_synthesis_recipes"
    get_entry_set_maker: GetEntrySetMaker = field(default_factory=GetEntrySetMaker)
    enumeration_maker: ReactionEnumerationMaker = field(
        default_factory=ReactionEnumerationMaker
    )
    calculate_selectivities_maker: CalculateSelectivitiesMaker = field(
        default_factory=CalculateSelectivitiesMaker
    )
    open_elem: Optional[Element] = None
    chempots: Optional[List[float]] = None
    use_basic_enumerators: bool = True
    use_minimize_enumerators: bool = True
    basic_enumerator_kwargs: dict = field(default_factory=dict)
    minimize_enumerator_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = (
            Composition(str(self.open_elem)).reduced_formula if self.open_elem else None
        )

    def make(  # type: ignore
        self,
        target_formula: str,
        added_elems: Optional[Collection[str]] = None,
        entries: Optional[GibbsEntrySet] = None,
    ):
        target_formula = Composition(target_formula).reduced_formula

        flow_name = f"Retrosynthesis: {target_formula}"

        if added_elems is None:
            added_elems = []
        else:
            flow_name = flow_name + f" (+ {'-'.join(sorted(added_elems))})"

        chemsys = "-".join(
            {str(e) for e in Composition(target_formula).elements}
            | {str(e) for e in added_elems}
        )

        jobs = []

        if entries is None:
            get_entry_set_maker = self.get_entry_set_maker.update_kwargs(
                {
                    "name": self.get_entry_set_maker.name
                    + f" ({chemsys}, T={self.get_entry_set_maker.temperature} K,"
                    f" +{round(self.get_entry_set_maker.e_above_hull, 3)} eV)"
                }
            )
            get_entry_set_job = get_entry_set_maker.make(chemsys)
            jobs.append(get_entry_set_job)
            entries = get_entry_set_job.output.entries

        targets = [target_formula]
        filter_by_chemsys = Composition(target_formula).chemical_system

        basic_enumerator_kwargs = self.basic_enumerator_kwargs.copy()
        minimize_enumerator_kwargs = self.minimize_enumerator_kwargs.copy()

        kwarg_update = dict(targets=targets, filter_by_chemsys=filter_by_chemsys)

        basic_enumerator_kwargs.update(kwarg_update)
        minimize_enumerator_kwargs.update(kwarg_update)

        enumerators = []

        if self.use_basic_enumerators:
            enumerators.append(
                BasicEnumerator(
                    targets=targets,
                    filter_by_chemsys=filter_by_chemsys,
                    **self.basic_enumerator_kwargs,
                )
            )
            if self.open_formula:
                enumerators.append(
                    BasicOpenEnumerator(
                        open_phases=[self.open_formula],
                        **self.basic_enumerator_kwargs,
                    )
                )
        if self.use_minimize_enumerators:
            enumerators.append(
                MinimizeGibbsEnumerator(
                    targets=targets,
                    filter_by_chemsys=filter_by_chemsys,
                    **self.minimize_enumerator_kwargs,
                )
            )

        enumeration_job = self.enumeration_maker.make(
            enumerators=enumerators, entries=entries
        )
        jobs.append(enumeration_job)

        base_rxn_set = enumeration_job.output.rxns
        base_calculate_selectivities_job = self.calculate_selectivities_maker.make(
            rxn_sets=[base_rxn_set],
            entries=get_entry_set_job.output.entries,
            target_formula=target_formula,
        )
        jobs.append(base_calculate_selectivities_job)

        if self.use_minimize_enumerators and self.open_elem and self.chempots:
            for chempot in self.chempots:
                subname = f"(open {str(self.open_elem)}, mu={chempot})"
                enumeration_maker = self.enumeration_maker.update_kwargs(
                    {"name": self.enumeration_maker.name + subname},
                    nested=False,
                )
                calculate_selectivities_maker = (
                    self.calculate_selectivities_maker.update_kwargs(
                        {
                            "chempot": chempot,
                            "open_elem": self.open_elem,
                            "name": self.calculate_selectivities_maker.name + subname,
                        },
                        nested=False,
                    )
                )
                enumerator = MinimizeGrandPotentialEnumerator(
                    open_elem=self.open_elem,
                    mu=chempot,
                    targets=targets,
                    filter_by_chemsys=filter_by_chemsys,
                    **self.minimize_enumerator_kwargs,
                )
                enumeration_job = enumeration_maker.make(
                    enumerators=[enumerator], entries=get_entry_set_job.output.entries
                )
                jobs.append(enumeration_job)

                calculate_selectivities_job = calculate_selectivities_maker.make(
                    rxn_sets=[base_rxn_set, enumeration_job.output.rxns],
                    entries=entries,
                    target_formula=target_formula,
                )

                jobs.append(calculate_selectivities_job)

        return Flow(jobs, name=flow_name)


@dataclass
class NetworkFlowMaker(Maker):
    name: str = "find_reaction_pathways"
    get_entry_set_maker: GetEntrySetMaker = field(default_factory=GetEntrySetMaker)
    enumeration_maker: ReactionEnumerationMaker = field(
        default_factory=ReactionEnumerationMaker
    )
    network_maker: NetworkMaker = field(default_factory=NetworkMaker)
    solver_maker: Optional[PathwaySolverMaker] = None
    open_elem: Optional[Element] = None
    chempots: Optional[List[float]] = None
    use_basic_enumerators: bool = True
    use_minimize_enumerators: bool = True
    basic_enumerator_kwargs: dict = field(default_factory=dict)
    minimize_enumerator_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = (
            Composition(str(self.open_elem)).reduced_formula if self.open_elem else None
        )

    def make(self, precursors, targets, entries=None):
        precursor_formulas = [Composition(f).reduced_formula for f in precursors]
        target_formulas = [Composition(f).reduced_formula for f in targets]

        flow_name = (
            f"Reaction Network analysis: {'-'.join(sorted(precursor_formulas))} ->"
            f" {'-'.join(sorted(target_formulas))}"
        )
        chemsys = "-".join(
            {
                str(e)
                for formula in precursor_formulas + target_formulas
                for e in Composition(formula).elements
            }
        )

        jobs = []

        if entries is None:
            get_entry_set_maker = self.get_entry_set_maker.update_kwargs(
                {
                    "name": self.get_entry_set_maker.name
                    + f" ({chemsys}, T={self.get_entry_set_maker.temperature} K,"
                    f" +{round(self.get_entry_set_maker.e_above_hull, 3)} eV)"
                }
            )
            get_entry_set_job = get_entry_set_maker.make(chemsys)
            jobs.append(get_entry_set_job)
            entries = get_entry_set_job.output.entries

        basic_enumerator_kwargs = self.basic_enumerator_kwargs.copy()
        minimize_enumerator_kwargs = self.minimize_enumerator_kwargs.copy()

        enumerators = []

        if self.use_basic_enumerators:
            enumerators.append(
                BasicEnumerator(
                    **self.basic_enumerator_kwargs,
                )
            )
            if self.open_formula:
                enumerators.append(
                    BasicOpenEnumerator(
                        open_phases=[self.open_formula],
                        **self.basic_enumerator_kwargs,
                    )
                )
        if self.use_minimize_enumerators:
            enumerators.append(
                MinimizeGibbsEnumerator(
                    **self.minimize_enumerator_kwargs,
                )
            )

        enumeration_job = self.enumeration_maker.make(
            enumerators=enumerators, entries=entries
        )
        jobs.append(enumeration_job)

        base_rxn_set = enumeration_job.output.rxns

        base_network_job = self.network_maker.make([base_rxn_set])
        jobs.append(base_network_job)

        if self.solver_maker:
            base_pathway_job = self.solver_maker.make(
                base_network_job.output.paths, entries=entries
            )
            jobs.append(base_pathway_job)

        if self.use_minimize_enumerators and self.open_elem and self.chempots:
            for chempot in self.chempots:
                subname = f"(open {str(self.open_elem)}, mu={chempot})"
                enumeration_maker = self.enumeration_maker.update_kwargs(
                    {"name": self.enumeration_maker.name + subname},
                    nested=False,
                )
                network_maker = self.network_maker.update_kwargs(
                    {"name": self.network_maker.name + subname},
                    nested=False,
                )
                if self.solver_maker:
                    solver_maker = self.solver_maker.update_kwargs(
                        {"name": self.solver_maker.name + subname},
                        nested=False,
                    )

                enumerator = MinimizeGrandPotentialEnumerator(
                    open_elem=self.open_elem,
                    mu=chempot,
                    **self.minimize_enumerator_kwargs,
                )
                enumeration_job = enumeration_maker.make(
                    enumerators=[enumerator], entries=entries
                )
                network_job = network_maker.make(
                    [base_rxn_set, enumeration_job.output.rxns]
                )
                jobs.extend([enumeration_job, network_job])
                if self.solver_maker:
                    pathway_job = solver_maker.make(
                        network_job.output.paths, entries=entries
                    )
                    jobs.append(pathway_job)

        return Flow(jobs, name=flow_name)
