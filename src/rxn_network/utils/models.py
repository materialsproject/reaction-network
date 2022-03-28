"""Core definition for various task and synthesis recipe documents"""
from datetime import datetime
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from pymatgen.core.composition import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core import CostFunction, Enumerator, Network
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet


class EnumeratorTask(BaseModel):
    """
    A single task object from the reaction enumerator workflow.
    """

    task_id: str = Field(None, description="The task id")
    task_label: str = Field(None, description="The name of the task.")
    last_updated: Optional[datetime] = Field(None, description="The last updated time")
    dir_name: str = Field(None, description="The directory name")
    rxns: Optional[ReactionSet] = Field(
        None, description="The reaction set. Optional to allow for storing in GridFS."
    )
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
    cost_function: CostFunction = Field(
        None, description="The cost function used to calculate the cost."
    )

    @classmethod
    def from_rxns_and_metadata(cls, rxns: ReactionSet, metadata: dict, **kwargs):
        """
        Create an EnumeratorTask from the reaction enumerator outputs.

        Args:
            rxns: the reaction set
            metadata: The metadata dictionary.
            **kwargs: Additional keyword arguments to pass to the EnumeratorTask constructor.
        """
        data = metadata.copy()
        data["rxns"] = rxns

        d = {k: v for k, v in data.items() if v is not None}

        return cls(**d, **kwargs)


class NetworkTask(BaseModel):
    """
    TODO: Finish model

    A single task object from the reaction network workflow.
    """

    task_id: str = Field(None, description="The task id")
    task_label: str = Field(None, description="The name of the task.")
    last_updated: Optional[datetime] = Field(None, description="The last updated time")
    dir_name: str = Field(None, description="The directory name")

    @classmethod
    def from_network_and_metadata(cls, network: Network, metadata: dict, **kwargs):
        """
        Create a NetworkTask from the reaction network workflow outputs.

        Args:
            network: the Network object
            metadata: The metadata dictionary.
            **kwargs: Additional keyword arguments to pass to the EnumeratorTask constructor.
        """
        data = metadata.copy()
        data["network"] = network

        d = {k: v for k, v in data.items() if v is not None}

        return cls(**d, **kwargs)


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

    @classmethod
    def from_computed_entry(cls, entry: ComputedEntry, **kwargs) -> "Phase":
        """
        Simple constructor for a phase from a pymatgen ComputedEntry object.
        """
        data: Dict[str, Any] = {}

        data["name"] = entry.name
        data["composition"] = entry.composition
        data["energy"] = entry.energy
        data["energy_per_atom"] = entry.energy_per_atom
        data["reduced_formula"] = entry.composition.reduced_formula
        data["temperature"] = getattr(entry, "temperature", None)
        data["entry_id"] = entry.entry_id
        data["e_above_hull"] = entry.data.get("e_above_hull", None)
        data["is_metastable"] = (
            data["e_above_hull"] > 0 if data["e_above_hull"] is not None else None
        )
        data["is_experimental"] = entry.is_experimental
        data["icsd_ids"] = entry.data.get("icsd_ids", None)
        d = {k: v for k, v in data.items() if v is not None}

        return cls(**d, **kwargs)


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

    chempot_distance: float = Field(
        None, description="Chemical potential distance score of the reaction"
    )
    c_score: float = Field(None, description="Competition score of the reaction")
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
    compositions: List[Composition] = Field(
        None, description="List of composition objects in the reaction equation."
    )
    coefficients: List[float] = Field(
        None,
        description="List of coefficients corresponding to compositions in reactin equation.",
    )

    @classmethod
    def from_computed_rxn(cls, rxn: ComputedReaction, **kwargs) -> "Reaction":
        """
        Simple constructor for a Reaction model from a ComputedReaction object.
        """
        data: Dict[str, Any] = {}

        data["rxn_str"] = str(rxn)
        data["reactants"] = [Phase.from_computed_entry(r) for r in rxn.reactant_entries]
        data["products"] = [Phase.from_computed_entry(p) for p in rxn.product_entries]
        data["elements"] = rxn.elements
        data["chemsys"] = rxn.chemical_system
        data["compositions"] = rxn.compositions
        data["coefficients"] = rxn.coefficients.tolist()

        d = {k: v for k, v in data.items() if v is not None}

        return cls(**d, **kwargs)


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

    @classmethod
    def from_computed_rxn(
        cls,
        rxn: ComputedReaction,
        cost: float,
        target: Composition,
        mu_func: Optional[str] = None,
        **kwargs
    ) -> "ComputedSynthesisRecipe":
        """
        Simple constructor for a Reaction model from a ComputedReaction object.
        """
        data: Dict[str, Any] = {}

        data["rxn"] = Reaction.from_computed_rxn(rxn)
        data["thermo"] = ReactionThermo(
            energy_per_atom=rxn.energy_per_atom,
            energy_uncertainty_per_atom=rxn.energy_uncertainty_per_atom,
        )
        data["selectivity"] = ReactionSelectivity(
            chempot_distance=rxn.data.get("chempot_distance"),
            c_score=rxn.data.get("c_score"),
            mu_func=mu_func,
        )
        data["heuristics"] = ReactionHeuristics(
            is_separable=rxn.is_separable(target),
            fraction_experimental=sum([e.is_experimental for e in rxn.entries])
            / len(rxn.entries),
        )
        data["cost"] = cost
        data["byproducts"] = sorted(
            [c.reduced_formula for c in (set(rxn.products) - {target})]
        )

        d = {k: v for k, v in data.items() if v is not None}

        return cls(**d, **kwargs)


class ComputedSynthesisRecipesDoc(BaseModel):
    """
    Model for a document containing computed synthesis recipes for one material
    """

    task_id: str = Field(None, description="Task ID")
    task_label: str = Field(None, description="The name of the task document")
    last_updated: Optional[datetime] = Field(None, description="The last updated time")
    recipes: List[ComputedSynthesisRecipe] = Field(
        None, description="List of computed synthesis recipes"
    )
    target_composition: Composition = Field(None, description="Target composition")
    target_formula: str = Field(None, description="The target formula")
    elements: List[Element] = Field(None, description="List of elements in the recipe")
    chemsys: str = Field(
        None, description="The total chemical system string (e.g., Fe-Li-O)."
    )
    added_elements: List[Element] = Field(None, description="List of elements added")
    added_chemsys: str = Field(
        None,
        description="The chemical system string of the added elements (e.g., Fe-Li-O)",
    )
    enumerators: List[Enumerator] = Field(
        None,
        description="A list of the enumerator objects used to calculate the reactions.",
    )
    cost_function: CostFunction = Field(
        None, description="The cost function used to calculate the cost."
    )
