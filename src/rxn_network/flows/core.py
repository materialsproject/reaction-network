"""Core flows for the reaction-network package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.core.composition import Element

from rxn_network.core import Composition
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator
from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator, MinimizeGrandPotentialEnumerator
from rxn_network.jobs.core import (
    CalculateCompetitionMaker,
    GetEntrySetMaker,
    NetworkMaker,
    PathwaySolverMaker,
    ReactionEnumerationMaker,
)
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from rxn_network.entries.entry_set import GibbsEntrySet

logger = get_logger(__name__)


@dataclass
class SynthesisPlanningFlowMaker(Maker):
    """Maker to create an inorganic synthesis planning workflow. This flow has three
    stages.

    Steps:
        1)  Entries are acquired via `GetEntrySetMaker`. This job both gets the computed
            entries from a databse (e.g., Materials Project) and processes them for use in
            the reaction network.
        2)  Reactions are enumerated via the provided `ReactionEnumerationMaker` (and
            associated enumerators). This computes the full reaction network so that
            selectivities can be calculated.
        3)  The competition of all synthesis reactions to the desired target is assessed via
            the `CalculateCompetitionMaker`.

    This flow also has the option to include an "open" element and a list of chempots.
    This will enumerate reactions at different conditions and evaluate their
    selectivities at those conditinons.

    This flow does not produce a specific output document. Instead, it is convenient to
    analyze output documents from each of the jobs in the flow based on the desired
    analysis. For the final "results", one should access the reaction set produced by
    the `CalculateCompetitionMaker` at the conditions of interest.

    If you use this code in your work, please cite the following work:

        McDermott, M. J. et al. Assessing Thermodynamic Selectivity of Solid-State Reactions for the Predictive
        Synthesis of Inorganic Materials. ACS Cent. Sci. (2023) doi:10.1021/acscentsci.3c01051.

    Args:
        name: Name of the flow. Automatically generated if not provided.
        get_entry_set_maker: `GetEntrySetMaker`used to create the job for acquiring
            entries. Automatically generated with default settings if not provided.
        enumeration_maker: `ReactionEnumerationMaker` used to create the reaction
            enumeration job. Automatically generated with default settings if not
            provided.
        calculate_competition_maker: `CalculateCompetitionMaker` used to create the
            selectivity analysis job. Automatically generated with default settings if
            not provided.
        open_elem: Optional element to use as the "open" element. If provided, the flow
            will  enumerate reactions at different chemical potentials of this element.
        chempots: List of chemical potentials to use for the "open" element. If
            provided, the flow will enumerate reactions at different chemical potentials
            of this element.
        use_basic_enumerators: Whether to use the `BasicEnumerator` and
            `BasicOpenEnumerator` enumerators in the enumeration job.
        use_minimize_enumerators: Whether to use the `MinimizeGibbsEnumerator` and the
            `MinimizeGrandPotentialEnumerator` enumerators in the enumeration job.
        basic_enumerator_kwargs: Keyword arguments to pass to the basic enumerators.
        minimize_enumerator_kwargs: Keyword arguments to pass to the minimize
            enumerators.
    """

    name: str = "synthesis_planning"
    get_entry_set_maker: GetEntrySetMaker = field(default_factory=GetEntrySetMaker)
    enumeration_maker: ReactionEnumerationMaker = field(default_factory=ReactionEnumerationMaker)
    calculate_competition_maker: CalculateCompetitionMaker = field(default_factory=CalculateCompetitionMaker)
    open_elem: Element | str | None = None
    chempots: list[float] | None = None
    use_basic_enumerators: bool = True
    use_minimize_enumerators: bool = True
    basic_enumerator_kwargs: dict = field(default_factory=dict)
    minimize_enumerator_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = Composition(str(self.open_elem)).reduced_formula if self.open_elem else None

    def make(  # type: ignore
        self,
        target_formula: str,
        added_elems: Collection[str] | None = None,
        entries: GibbsEntrySet | None = None,
    ):
        """Returns a flow used for planning optimal synthesis recipes to a specified
        target.

        Args:
            target_formula: The chemical formula of a target phase (e.g., "BaTiO3").
            added_elems: An optional list of additional elements to consider (e.g.,
                ["C", "H"]). Defaults to None.
            entries: An optional provided set of entries to enumerate from. If entries
                are not provided, then they will be acquired from a database (e.g.,
                Materials Project) and processed using the GetEntrySetMaker.

        """
        target_formula = Composition(target_formula).reduced_formula

        flow_name = f"Synthesis planning: {target_formula}"

        if added_elems is None:
            added_elems = []
        else:
            flow_name = flow_name + f" (+ {'-'.join(sorted(added_elems))})"

        flow_name = flow_name + f", T={self.get_entry_set_maker.temperature} K"

        chemsys = "-".join(
            sorted({str(e) for e in Composition(target_formula).elements} | {str(e) for e in added_elems})
        )

        jobs = []

        if entries is None:
            get_entry_set_maker = self.get_entry_set_maker.update_kwargs(
                {
                    "name": self.get_entry_set_maker.name + f" ({chemsys}, T={self.get_entry_set_maker.temperature} K,"
                    f" +{round(self.get_entry_set_maker.e_above_hull, 3)} eV)",
                    "formulas_to_include": list({*self.get_entry_set_maker.formulas_to_include, target_formula}),
                }
            )
            get_entry_set_job = get_entry_set_maker.make(chemsys)
            jobs.append(get_entry_set_job)
            entries = get_entry_set_job.output.entries

        targets = [target_formula]
        filter_by_chemsys = Composition(target_formula).chemical_system

        basic_enumerator_kwargs = self.basic_enumerator_kwargs.copy()
        minimize_enumerator_kwargs = self.minimize_enumerator_kwargs.copy()

        kwarg_update = {"targets": targets, "filter_by_chemsys": filter_by_chemsys}

        basic_enumerator_kwargs.update(kwarg_update)
        minimize_enumerator_kwargs.update(kwarg_update)

        enumerators = []

        if self.use_basic_enumerators:
            enumerators.append(
                BasicEnumerator(
                    filter_by_chemsys=filter_by_chemsys,
                    **self.basic_enumerator_kwargs,
                )
            )
        if self.use_minimize_enumerators:
            enumerators.append(
                MinimizeGibbsEnumerator(
                    filter_by_chemsys=filter_by_chemsys,
                    **self.minimize_enumerator_kwargs,
                )
            )

        enumeration_job = self.enumeration_maker.make(enumerators=enumerators, entries=entries)
        jobs.append(enumeration_job)

        base_rxn_set = enumeration_job.output.rxns
        calculate_competition_maker = self.calculate_competition_maker

        base_calculate_competition_job = calculate_competition_maker.make(
            rxn_sets=[base_rxn_set],
            entries=entries,
            target_formula=target_formula,
        )
        jobs.append(base_calculate_competition_job)

        if self.open_elem and self.chempots:
            for chempot in self.chempots:
                subname = f"(open {self.open_elem!s}, mu={chempot})"
                enumeration_maker = self.enumeration_maker.update_kwargs(
                    {"name": self.enumeration_maker.name + subname},
                    nested=False,
                )
                calculate_competition_maker = calculate_competition_maker.update_kwargs(
                    {
                        "chempot": chempot,
                        "open_elem": self.open_elem,
                        "name": self.calculate_competition_maker.name + subname,
                    },
                    nested=False,
                )

                open_enumerators = []

                if self.use_basic_enumerators:
                    open_enumerators.append(
                        BasicOpenEnumerator(
                            open_phases=[self.open_formula],
                            **self.basic_enumerator_kwargs,
                        )
                    )
                if self.use_minimize_enumerators:
                    open_enumerators.append(
                        MinimizeGrandPotentialEnumerator(
                            open_elem=self.open_elem,
                            mu=chempot,
                            filter_by_chemsys=filter_by_chemsys,
                            **self.minimize_enumerator_kwargs,
                        )
                    )
                enumeration_job = enumeration_maker.make(
                    enumerators=open_enumerators,
                    entries=entries,
                )
                jobs.append(enumeration_job)

                calculate_competition_job = calculate_competition_maker.make(
                    rxn_sets=[base_rxn_set, enumeration_job.output.rxns],
                    entries=entries,
                    target_formula=target_formula,
                )

                jobs.append(calculate_competition_job)

        return Flow(jobs, name=flow_name)


@dataclass
class NetworkFlowMaker(Maker):
    """Maker to create a chemical reaction network and perform (balanced) pathfinding on
    the network.

    This flow has four stages:

    1)  Entries are acquired via `GetEntrySetMaker`. This job both gets the computed
        entries from a databse (e.g., Materials Project) and processes them for use in
        the reaction network.
    2)  Reactions are enumerated via the provided `ReactionEnumerationMaker` (and
        associated enumerators).
    3)  The network is created using `NetworkMaker` and basic paths are found
        (k-shortest paths to each target).
    4)  The final balanced reaction pathways are produced using the `SolverMaker`.

    If you use this code in your own work, please consider citing this paper:

        McDermott, M. J.; Dwaraknath, S. S.; Persson, K. A. A Graph-Based Network for
        Predicting Chemical Reaction Pathways in Solid-State Materials Synthesis. Nature
        Communications 2021, 12 (1), 3097. https://doi.org/10.1038/s41467-021-23339-x.

    Args:
        name: The name of the network flow. Automatically assigned if not provided.
        get_entry_set_maker: `GetEntrySetMaker`used to create the job for acquiring
            entries. Automatically generated with default settings if not provided.
        enumeration_maker: `ReactionEnumerationMaker` used to create the reaction
            enumeration job. Automatically generated with default settings if not
            provided.
        network_maker: `NetworkMaker` used to create the reaction network from sets of
            reactions. Also identifies basic reaction pathways. Automatically generated
            with default settings if not provided.
        solver_maker: `PathwaySolverMaker` used to find balanced reaction pathways from
            set of pathways emerging from pathfinding. Automatically generated with
            default settings if not provided.
        open_elem: Optional element to use as the "open" element. If provided, the flow
            will enumerate reactions at different chemical potentials of this element.
        chempots: List of chemical potentials to use for the "open" element. If
            provided, the flow will enumerate reactions at different chemical potentials
            of this element.
        use_basic_enumerators: Whether to use the `BasicEnumerator` and
            `BasicOpenEnumerator` enumerators in the enumeration job.
        use_minimize_enumerators: Whether to use the `MinimizeGibbsEnumerator` and the
            `MinimizeGrandPotentialEnumerator` enumerators in the enumeration job.
        basic_enumerator_kwargs: Keyword arguments to pass to the basic enumerators.
        minimize_enumerator_kwargs: Keyword arguments to pass to the minimize
            enumerators.
    """

    name: str = "find_reaction_pathways"
    get_entry_set_maker: GetEntrySetMaker = field(default_factory=GetEntrySetMaker)
    enumeration_maker: ReactionEnumerationMaker = field(default_factory=ReactionEnumerationMaker)
    network_maker: NetworkMaker = field(default_factory=NetworkMaker)
    solver_maker: PathwaySolverMaker | None = None
    open_elem: Element | None = None
    chempots: list[float] | None = None
    use_basic_enumerators: bool = True
    use_minimize_enumerators: bool = True
    basic_enumerator_kwargs: dict = field(default_factory=dict)
    minimize_enumerator_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.open_elem = Element(self.open_elem) if self.open_elem else None
        self.open_formula = Composition(str(self.open_elem)).reduced_formula if self.open_elem else None

    def make(self, precursors: Iterable[str], targets: Iterable[str], entries: GibbsEntrySet | None = None):
        """Returns a flow used for finding reaction pathways between precursors and targets.

        Args:
            precursors: precursor formulas
            targets: target formulas
            entries: Optional entry set. If not provided, entries will be automatically acquired from Materials Project.
                Defaults to None.

        Returns:
            _description_
        """
        precursor_formulas = [Composition(f).reduced_formula for f in precursors]
        target_formulas = [Composition(f).reduced_formula for f in targets]

        flow_name = (
            f"Reaction network analysis: {'-'.join(sorted(precursor_formulas))} ->"
            f" {'-'.join(sorted(target_formulas))}"
        )
        chemsys = "-".join(
            {str(e) for formula in precursor_formulas + target_formulas for e in Composition(formula).elements}
        )

        jobs = []

        if entries is None:
            get_entry_set_maker = self.get_entry_set_maker.update_kwargs(
                {
                    "name": self.get_entry_set_maker.name + f" ({chemsys}, T={self.get_entry_set_maker.temperature} K,"
                    f" +{round(self.get_entry_set_maker.e_above_hull, 3)} eV)",
                    "formulas_to_include": list(
                        set(self.get_entry_set_maker.formulas_to_include + precursor_formulas + target_formulas)
                    ),
                }
            )
            get_entry_set_job = get_entry_set_maker.make(chemsys)
            jobs.append(get_entry_set_job)
            entries = get_entry_set_job.output.entries

        self.basic_enumerator_kwargs.copy()
        self.minimize_enumerator_kwargs.copy()

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

        enumeration_job = self.enumeration_maker.make(enumerators=enumerators, entries=entries)
        jobs.append(enumeration_job)

        base_rxn_set = enumeration_job.output.rxns

        base_network_job = self.network_maker.make([base_rxn_set])
        jobs.append(base_network_job)

        if self.solver_maker:
            base_pathway_job = self.solver_maker.make(
                base_network_job.output.paths,
                entries=base_network_job.output.network.entries,
            )
            jobs.append(base_pathway_job)

        if self.use_minimize_enumerators and self.open_elem and self.chempots:
            for chempot in self.chempots:
                subname = f"(open {self.open_elem!s}, mu={chempot})"
                enumeration_maker = self.enumeration_maker.update_kwargs(
                    {"name": self.enumeration_maker.name + subname},
                    nested=False,
                )
                network_maker = self.network_maker.update_kwargs(
                    {
                        "name": self.network_maker.name + subname,
                        "chempot": chempot,
                        "open_elem": self.open_elem,
                    },
                    nested=False,
                )
                if self.solver_maker:
                    solver_maker = self.solver_maker.update_kwargs(
                        {
                            "name": self.solver_maker.name + subname,
                            "chempot": chempot,
                            "open_elem": self.open_elem,
                        },
                        nested=False,
                    )

                enumerator = MinimizeGrandPotentialEnumerator(
                    open_elem=self.open_elem,
                    mu=chempot,
                    **self.minimize_enumerator_kwargs,
                )
                enumeration_job = enumeration_maker.make(enumerators=[enumerator], entries=entries)
                network_job = network_maker.make([base_rxn_set, enumeration_job.output.rxns])
                jobs.extend([enumeration_job, network_job])
                if self.solver_maker:
                    pathway_job = solver_maker.make(
                        network_job.output.paths,
                        entries=network_job.output.network.entries,
                    )
                    jobs.append(pathway_job)

        return Flow(jobs, name=flow_name)
