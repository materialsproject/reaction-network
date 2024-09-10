"""Core definition for various task and synthesis recipe documents."""

from typing import Optional

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

    task_label: Optional[str] = Field(None, description="The name of the task.")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    entries: GibbsEntrySet = Field(description="The entry set object.")
    e_above_hull: Optional[float] = Field(None, description="The e_above_hull cutoff.")
    include_polymorphs: bool = Field(
        default=False, description="Whether to include metastable polymorphs in the entry set."
    )
    formulas_to_include: Optional[list[str]] = Field(
        None, description="The required formulas to include during construciton."
    )


class EnumeratorTaskDocument(BaseModel):
    """A single task object from the reaction enumerator workflow."""

    task_label: Optional[str] = Field(None, description="The name of the task.")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    rxns: ReactionSet = Field(description="The reaction set.")
    targets: Optional[list[str]] = Field(None, description="The target formulas used in the enumerator(s).")
    elements: Optional[list[Element]] = Field(None, description="The elements of the total chemical system")
    chemsys: Optional[str] = Field(None, description="The total chemical system string (e.g., Fe-Li-O).")
    added_elements: Optional[list[Element]] = Field(
        None, description="The elements added beyond the elements of the target(s)."
    )
    added_chemsys: Optional[str] = Field(None, description="The chemical system of the added elements")
    enumerators: Optional[list[Enumerator]] = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )


class NetworkTaskDocument(BaseModel):
    """The calculation output from the NetworkMaker workflow.

    Optionally includes unbalanced paths found during pathfinding.
    """

    task_label: Optional[str] = Field(None, description="The name of the task.")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    network: Network = Field(description="The reaction network")
    paths: Optional[PathwaySet] = Field(None, description="The (simple/unbalanced) reaction pathways")
    k: Optional[int] = Field(None, description="The number of paths solved for, if any.")
    precursors: Optional[list[str]] = Field(None, description="The precursor formulas.")
    targets: Optional[list[str]] = Field(None, description="The target formulas.")


class PathwaySolverTaskDocument(BaseModel):
    """The calculation output from the PathwaySolverMaker workflow. Contains the pathway
    solver object and its calculated balanced pathways.
    """

    task_label: Optional[str] = Field(None, description="The name of the task.")
    last_updated: str = Field(
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
