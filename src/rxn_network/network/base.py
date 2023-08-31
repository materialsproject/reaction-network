"""
Basic interface for a reaction network and its graph.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Iterable

from monty.json import MontyDecoder, MSONable
from rustworkx import PyDiGraph

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
        """Returns MSONable dict for serialization. See monty package for more
        information."""
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
    def precursors(self) -> set[Entry] | None:
        """The phases used as precursors in the network"""
        return self._precursors

    @property
    def target(self) -> Entry | None:
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


class Graph(PyDiGraph):
    """
    Thin wrapper around rx.PyDiGraph to allow for serialization and optimized database
    storage.
    """

    def as_dict(self) -> dict:
        """
        Represents the PyDiGraph object as a serializable dictionary.

        See monty package (MSONable) for more information.
        """
        d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__}

        d["nodes"] = [n.as_dict() for n in self.nodes()]  # type: ignore
        d["node_indices"] = list(self.node_indices())  # type: ignore
        d["edges"] = [
            (*e, obj.as_dict() if hasattr(obj, "as_dict") else obj)  # type: ignore
            for e, obj in zip(self.edge_list(), self.edges())
        ]

        return d

    @classmethod
    def from_dict(cls, d: dict) -> Graph:
        """
        Instantiates a Graph object from a dictionary.

        See as_dict() and monty package (MSONable) for more information.
        """
        nodes = MontyDecoder().process_decoded(d["nodes"])
        node_indices = MontyDecoder().process_decoded(d["node_indices"])
        edges = [(e[0], e[1], MontyDecoder().process_decoded(e[2])) for e in d["edges"]]

        nodes = dict(zip(nodes, node_indices))

        graph = cls()
        new_indices = graph.add_nodes_from(list(nodes.keys()))
        mapping = {nodes[node]: idx for idx, node in zip(new_indices, nodes.keys())}

        new_mapping = []
        for edge in edges:
            new_mapping.append((mapping[edge[0]], mapping[edge[1]], edge[2]))

        graph.add_edges_from(new_mapping)

        return graph

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f" with {self.num_nodes()} nodes and {self.num_edges()} edges"
        )
