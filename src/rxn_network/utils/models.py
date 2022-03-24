"""Core definition for various task and synthesis recipe documents"""
import os
from enum import Enum
from typing import Any, List, Optional

from monty.serialization import loadfn
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from pymatgen.core.composition import Composition, Element

from rxn_network.core.enumerator import Enumerator


class EnumeratorTask(BaseModel):
    """
    A single task object from the reaction enumerator workflow.
    """

    name: str = Field(None, description="The name of the task.")
    task_id: str = Field(None, description="The task id")
    state: Enum = Field(None, description="The task state")
    last_updated: Optional[str] = Field(None, description="The last updated time")
    dir_name: str = Field(None, description="The directory name")
    rxns_fs_id: str = Field(
        None, description="The GridFS ID referencing the reaction enumerator output"
    )
    elements: List[Element] = Field(
        None, description="The elements of the total chemical system"
    )
    chemsys: str = Field(
        None, description="The total chemical system string (e.g., Fe-Li-O)."
    )
    added_chemsys: str = Field(
        None, description="The chemical system of the added elements"
    )
    enumerators: List[Enumerator] = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )
    cost_function: str = Field(
        None, description="The cost function used to calculate the cost."
    )

    @classmethod
    def from_files(cls, rxns_fn, metadata_fn, **kwargs):
        dir_name = os.path.abspath(os.getcwd())

        rxns = (loadfn(rxns_fn),)
        metadata = loadfn(metadata_fn)

        d = {"rxns": rxns, "metadata": metadata, "dir_name": dir_name}
        return cls(**d, **kwargs)


class NetworkTask(BaseModel):
    """
    A single task object from the reaction network workflow.
    """

    task_id: str = Field(..., description="The task id")
    state: Enum = Field(..., description="The task state")
    data: Optional[Any] = Field(None, description="The task data")
    last_updated: Optional[str] = Field(None, description="The last updated time")

    @classmethod
    def from_files(cls, rxn_file, metadata_file):
        return cls


class Phase(BaseModel):
    """
    A phase of a synthesis recipe.
    """

    name: str = Field(None, description="Name of the phase.")
    composition: Composition = Field(
        None, description="The composition object for the phase"
    )
    energy: float = Field(None, description="The energy of the phase")
    energy_per_atom: float = Field(
        None, description="The energy per atom of the phase."
    )
    reduced_formula: str = Field(None, description="The reduced formula of the phase.")
    temperature: float = Field(
        None, description="The temperature at which the energy was calculated."
    )
    entry_id: str = Field(
        None, description="The entry id (usually mp-id) of the phase."
    )
    e_above_hull: float = Field(
        None, description="The calculated e above hull of the phase"
    )
    is_metastable: bool = Field(
        None, description="Whether the phase has a positive e above hull."
    )
    is_experimental: bool = Field(
        None,
        description="Whether the phase has been observed experimentally in the literature.",
    )
    icsd_ids: List[int] = Field(None, description="A list of ICSD ids for the phase.")


class ReactionThermo(BaseModel):
    """
    A class for storing the reaction energetics and thermodynamics related parameters.
    """

    energy_per_atom: float = Field(None, description="Reaction energy in eV/atom")
    energy_uncertainty_per_atom: float = Field(
        None, description="Reaction energy uncertainty in eV/atom"
    )


class ReactionSelectivity(BaseModel):
    """
    A class for storing the reaction selectivity related parameters (e.g., chemical
    potential distance)
    """

    chempot_distance: float = Field(None, description="Chemical potential distance")
    c_score: float = Field(None, description="Competitiveness Score")
    mu_func: str = Field(
        None, description="Name of function used to calculate chempot_distance metric"
    )


class ReactionHeuristics(BaseModel):
    """
    A class for storing user-defined heuristics about the reaction (e.g., products separable)
    """

    is_separable: bool = Field(
        None, description="Whether reaction products are separable"
    )
    fraction_experimental: float = Field(None, description="")


class Reaction(BaseModel):
    """
    A class to represent a computed reaction.
    """

    rxn_str: str = Field(None, description="String of balanced reaction")
    reactants: List[Phase] = Field(None, description="List of reactant phases")
    products: List[Phase] = Field(None, description="List of product phases")
    elements: List[Element] = Field(
        None, description="List of elements in the reaction"
    )
    chemsys: str = Field(None, description="The chemical system string (e.g., Fe-Li-O)")


class ComputedSynthesisRecipe(BaseModel):
    """
    Model for a document containing synthesis description data
    """

    rxn: Reaction = Field(None, description="Balanced reaction")
    thermo: ReactionThermo = Field(None, description="Reaction thermodynamics data")
    selectivity: ReactionSelectivity = Field(
        None, description="Reaction selectivity data"
    )
    heuristics: ReactionHeuristics = Field(None, description="Reaction heuristics data")
    cost: float = Field(None, description="Calculated cost of the reaction")
    byproducts: List[str] = Field(None, description="List of byproduct formulas")


class ComputedSynthesisRecipesDoc(BaseModel):
    """
    Model for a document containing computed synthesis recipes for one material
    """

    rxns: List[ComputedSynthesisRecipe] = Field(
        None, description="List of computed synthesis recipes"
    )
    target_composition: Composition = Field(None, description="Target composition")
    target_formula: str = Field(None, description="The target formula")
    added_elems: List[Element] = Field(None, description="List of elements added")
    chemsys: str = Field(
        None, description="The total chemical system string (e.g., Fe-Li-O)."
    )
    enumerators: List[Enumerator] = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )
    cost_function: str = Field(
        None, description="The cost function used to calculate the cost."
    )
