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
    GetEntrySetMaker,
    NetworkMaker,
    PathwaySolverMaker,
    ReactionEnumerationMaker,
)
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rxn_network.entries.entry_set import GibbsEntrySet

logger = get_logger(__name__)


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
