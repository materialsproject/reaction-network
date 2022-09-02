"""Core definition for various task and synthesis recipe documents"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from monty.serialization import loadfn
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.core.enumerator import Enumerator
from rxn_network.core.network import Network
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.utils.funcs import datetime_str


class EntrySetDocument(BaseModel):

    task_label: str = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    entries: GibbsEntrySet = Field(description="The entry set")
    e_above_hull: float = Field(None, description="The e_above_hull cutoff")
    include_polymorphs: bool = Field(False, description="Whether to include polymorphs")
    formulas_to_include: Optional[List[str]] = Field(
        None, description="The formulas to include"
    )


class EnumeratorTaskDocument(BaseModel):
    """
    A single task object from the reaction enumerator workflow.
    """

    task_label: str = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    rxns: ReactionSet = Field(description="The reaction set.")
    targets: List[str] = Field(
        None, description="The target formulas used in the enumerator(s)."
    )
    elements: List[Element] = Field(
        None, description="The elements of the total chemical system"
    )
    chemsys: str = Field(
        None, description="The total chemical system string (e.g., Fe-Li-O)."
    )
    added_elements: List[Element] = Field(
        None, description="The elements added beyond the elements of the target(s)."
    )
    added_chemsys: str = Field(
        None, description="The chemical system of the added elements"
    )
    enumerators: List[Enumerator] = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )


class SelectivitiesTaskDocument(BaseModel):
    task_label: str = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
    rxns: ReactionSet = Field(description="The reaction set.")
    target_formula: str = Field(
        description="The reduced chemical formula of the target material."
    )
    open_elem: Element = Field(None, description="The open element")
    chempot: float = Field(
        None, description="The chemical potential of the open element"
    )
    calculate_selectivities: bool = Field(
        None, description="Whether to calculate selectivities"
    )
    calculate_chempot_distances: bool = Field(
        None, description="Whether to calculate chempot distances"
    )
    temp: float = Field(None, description="The temperature in K")
    batch_size: int = Field(None, description="The batch size for the reaction set")
    cpd_kwargs: Dict[str, Any] = Field(
        None, description="The kwargs for ChempotDistanceCalculator"
    )


class NetworkTaskDocument(BaseModel):
    """
    TODO: Finish model

    A single task object from the reaction network workflow.
    """

    task_label: str = Field(None, description="The name of the task.")
    last_updated: datetime = Field(
        default_factory=datetime_str,
        description="Timestamp of when the document was last updated.",
    )
