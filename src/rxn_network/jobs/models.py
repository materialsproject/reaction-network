"""Core definition for various task and synthesis recipe documents"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.core.enumerator import Enumerator
from rxn_network.core.network import Network
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet


class EnumeratorTaskDocument(BaseModel):
    """
    A single task object from the reaction enumerator workflow.
    """

    task_id: int = Field(None, description="The task id")
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


class NetworkTaskDocument(BaseModel):
    """
    TODO: Finish model

    A single task object from the reaction network workflow.
    """

    task_id: int = Field(None, description="The task id")
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