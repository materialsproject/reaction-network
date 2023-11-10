"""Core definition for various task and synthesis recipe documents."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pymatgen.core.composition import Element

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.base import Enumerator
from rxn_network.network.base import Network
from rxn_network.pathways.pathway_set import PathwaySet
from rxn_network.pathways.solver import Solver
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import datetime_str


class EntrySetDocument(BaseModel):
    """A single entry set object as produced by the GetEntrySet job."""

    task_label: str | None = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    entries: GibbsEntrySet = Field(description="The entry set object.")
    e_above_hull: float | None = Field(None, description="The e_above_hull cutoff.")
    include_polymorphs: bool = Field(False, description="Whether to include metastable polymorphs in the entry set.")
    formulas_to_include: list[str] | None = Field(
        None, description="The required formulas to include during construciton."
    )


class EnumeratorTaskDocument(BaseModel):
    """A single task object from the reaction enumerator workflow."""

    task_label: str | None = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    rxns: ReactionSet = Field(description="The reaction set.")
    targets: list[str] | None = Field(None, description="The target formulas used in the enumerator(s).")
    elements: list[Element] | None = Field(None, description="The elements of the total chemical system")
    chemsys: str | None = Field(None, description="The total chemical system string (e.g., Fe-Li-O).")
    added_elements: list[Element] | None = Field(
        None, description="The elements added beyond the elements of the target(s)."
    )
    added_chemsys: str | None = Field(None, description="The chemical system of the added elements")
    enumerators: list[Enumerator] | None = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )


class CompetitionTaskDocument(BaseModel):
    """A document containing the reactions and their selectivities as created by the
    CalculateCompetitionMaker job.
    """

    task_label: str | None = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    rxns: ReactionSet = Field(
        description=(
            "The reaction set, where thereactions have calculated competition"
            " information stored in their data attribute."
        )
    )
    target_formula: str = Field(description="The reduced chemical formula of the target material.")
    open_elem: Element | None = Field(None, description="The open element (if any).")
    chempot: float | None = Field(None, description="The chemical potential of the open element.")
    added_elements: list[Element] | None = Field(
        None, description="The elements added beyond the elements of the target(s)."
    )
    added_chemsys: str | None = Field(None, description="The chemical system of the added elements.")
    calculate_competition: bool | None = Field(None, description="Whether to calculate competition scores.")
    calculate_chempot_distances: bool | None = Field(
        None, description="Whether to calculate chemical potential distances."
    )
    temp: float | None = Field(
        None,
        description=("The temperature in K used to determine the primary competition weightings."),
    )
    batch_size: int | None = Field(None, description="The batch size for the reaction set")
    cpd_kwargs: dict[str, Any] | None = Field(None, description="The kwargs for ChempotDistanceCalculator.")


class NetworkTaskDocument(BaseModel):
    """The calculation output from the NetworkMaker workflow.

    Optionally includes unbalanced paths found during pathfinding.
    """

    task_label: str | None = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    network: Network = Field(description="The reaction network")
    paths: PathwaySet | None = Field(None, description="The (simple/unbalanced) reaction pathways")
    k: int | None = Field(None, description="The number of paths solved for, if any.")
    precursors: list[str] | None = Field(None, description="The precursor formulas.")
    targets: list[str] | None = Field(None, description="The target formulas.")


class PathwaySolverTaskDocument(BaseModel):
    """The calculation output from the PathwaySolverMaker workflow. Contains the pathway
    solver object and its calculated balanced pathways.
    """

    task_label: str | None = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    solver: Solver = Field(description="The pathway solver used to calculate pathways.")
    balanced_paths: PathwaySet = Field(description="The balanced reaction pathways.")
    precursors: list[str] = Field(description="The precursor compositions.")
    targets: list[str] = Field(description="The target compositions.")
    net_rxn: ComputedReaction = Field(description="The net reaction used for pathway solving.")
    max_num_combos: int = Field(description=("The maximum number of combinations to consider in the pathway solver."))
    find_intermediate_rxns: bool = Field(description="Whether to find reactions from intermediate compositions.")
    intermediate_rxn_energy_cutoff: float = Field(
        description="The mximum energy cutoff for filtering intermediate reactions."
    )
    use_basic_enumerator: bool = Field(description="Whether to use the basic enumerators in path solving.")
    use_minimize_enumerator: bool = Field(description="Whether to use the minimize enumerators in path solving.")
    filter_interdependent: bool = Field(description="Whether to filter out interdependent pathway.s")
