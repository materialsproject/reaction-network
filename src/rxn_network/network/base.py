"""
Basic interface for a reaction network and its graph.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Iterable

from monty.json import MontyDecoder, MSONable

from rxn_network.entries.entry_set import GibbsEntrySet

if TYPE_CHECKING:
    from pymatgen.entries import Entry

    from rxn_network.costs.base import CostFunction
    from rxn_network.pathways.base import Pathway
    from rxn_network.reactions.reaction_set import ReactionSet


class Network(MSONable, metaclass=ABCMeta):
    """
    Base definition for a reaction network.
    """

    def __init__(
        self,
        rxns: ReactionSet,
        cost_function: CostFunction,
    ):
        self.rxns = rxns
        self.cost_function = cost_function

        self.entries = GibbsEntrySet(rxns.entries)
        self.entries.build_indices()

        self._precursors = None
        self._target = None
        self._g = None

    @abstractmethod
    def build(self) -> None:
        """Construct the network in-place from the supplied enumerators"""

    @abstractmethod
    def find_pathways(self, target, k) -> list[Pathway]:
        """Find reaction pathways"""

    @abstractmethod
    def set_precursors(self, precursors: Iterable[Entry | str]) -> None:
        """Set the phases used as precursors in the network (in-place)"""

    @abstractmethod
    def set_target(self, target: Entry | str) -> None:
        """Set the phase used as a target in the network (in-place)"""

    def as_dict(self) -> dict:
        """Returns MSONable dict for serialization. See monty package for more nformation."""
        d = super().as_dict()
        d["precursors"] = list(self.precursors) if self.precursors else None
        d["target"] = self.target
        d["graph"] = self.graph.as_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Network:
        """Instantiate object from MSONable dict. See monty package for more
        information."""
        precursors = d.pop("precursors", None)
        target = d.pop("target", None)
        graph = d.pop("graph", None)

        network = super().from_dict(d)
        network._precursors = precursors  # pylint: disable=protected-access
        network._target = target  # pylint: disable=protected-access
        network._g = MontyDecoder().process_decoded(  # pylint: disable=protected-access
            graph
        )

        return network

    @property
    def precursors(self) -> set[Entry]:
        """The phases used as precursors in the network"""
        return self._precursors

    @property
    def target(self) -> Entry:
        """The phase used as a target in the network"""
        return self._target

    @property
    def graph(self):
        """Returns the network's Graph object"""
        return self._g

    @property
    def chemsys(self) -> str:
        """A string representing the chemical system (elements) of the network"""
        return "-".join(sorted(self.entries.chemsys))

    def __repr__(self) -> str:
        return (
            "Reaction network for chemical system: "
            f"{self.chemsys}, "
            f"with graph: {str(self.graph)}"
        )

    def __str__(self) -> str:
        return self.__repr__()
