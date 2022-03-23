""" Core definition of a SynthesisRecipe Document """

from enum import Enum
from typing import List, Optional, Any

from pydantic import BaseModel, Field

from pymatgen.core.composition import Composition

class EnumeratorTask(BaseModel):
    """
    A single task object from the reaction enumerator workflow.
    """

    task_id: str = Field(..., description="The task id")
    state: Enum = Field(..., description="The task state")
    data: Optional[Any] = Field(None, description="The task data")
    last_updated: Optional[str] = Field(None, description="The last updated time")


class Phase(BaseModel):
    """
    A phase of a synthesis recipe.
    """
    name: str = Field(None, "Name of the phase.")
    composition: Composition = Field(None, "The composition object for the phase")
    energy: float = Field(None, "The energy of the phase")
    energy_per_atom: float = Field(None, "The energy per atom of the phase")
    reduced_formula: str = Field(None, "The reduced formula of the phase")
    temperature: float = Field(None, "The temperature at which the energy was calculated.")
    entry_id: str = Field(None, "The entry id (usually mp-id) of the phase.")
    e_above_hull: float = Field(None, "The calculated e above hull of the phase")
    is_metastable: bool = Field(None, "Whether the phase has a positive e above hull.")
    is_experimental: bool = Field(None, "Whether the phase has been observed experimentally in the literature.")
    icsd_ids: List[int] = Field(None, "A list of ICSD ids for the phase.")

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
    chempot_distance: float = Field(None, "Chemical potential distance")
    c_score: float = Field(None, description="Competitiveness Score")
    mu_func: str = Field(None, "Name of function used to calculate chempot_distance metric")

class ReactionHeuristics(BaseModel):
    """
    A class for storing user-defined heuristics about the reaction (e.g., products separable)
    """
    is_separable: bool = Field(None, description="Whether reaction products are separable")
    fraction_experimental: float = Field(None, description="")

class Reaction(BaseModel):
    """
    A class to represent a computed reaction.
    """
    rxn_str: str = Field(None, description="String of balanced reaction")
    reactants: List[Phase] = Field(None, description="List of reactant phases")
    products: List[Phase] = Field(None, description="List of product phases")
    elements:
    chemsys:

class ComputedSynthesisRecipe(BaseModel):
    """
    Model for a document containing synthesis description data
    """

    rxn: Reaction = Field(None, description="Balanced reaction")
    thermo: ReactionThermo = Field(None, description="Reaction thermodynamics data")
    selectivity: ReactionSelectivity = Field(None, description="Reaction selectivity data")
    heuristics: ReactionHeuristics = Field(None, description="Reaction heuristics data")
    cost: float = Field(None, "Calculated cost of the reaction")
    byproducts: List[str] = Field(None, "List of byproduct formulas")

class ComputedSynthesisRecipesDoc(BaseModel):
    """
    Model for a document containing computed synthesis recipes for one material
    """

    rxns: List[ComputedSynthesisRecipe] = Field(
        None, description="List of computed synthesis recipes"
    )
    target_composition: Composition = Field(
    target_formula:
    added_elems:
    chemsys:
    enumerators:
    cost_function:

    @classmethod
    def from_rxns_and_metadata(cls, rxns, metadata):
        return cls