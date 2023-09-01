"""
Entry objects used in a Network. These network entry objects hold multiple entries and
can be used as data for a node in the graph.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Collection

from monty.json import MSONable
from monty.serialization import MontyDecoder

if TYPE_CHECKING:
    from pymatgen.core.periodic_table import Element
    from pymatgen.entries import Entry


class NetworkEntryType(Enum):
    """Describes the Network Entry Type"""

    Precursors = auto()
    Reactants = auto()
    Products = auto()
    Target = auto()
    Dummy = auto()


class NetworkEntry(MSONable):
    """
    Helper class for describing combinations of ComputedEntry-like objects in context
    of a reaction network. This entry will represent a node in the network.
    """

    def __init__(self, entries: Collection[Entry], description: NetworkEntryType):
        """
        Args:
            entries: Collection of Entry-like objects
            description: Node type (e.g., Precursors, Target... see NetworkEntryType
                class)
        """
        self._entries = set(entries)
        self._elements = sorted(
            list({elem for entry in entries for elem in entry.composition.elements})
        )
        self._chemsys = "-".join([str(e) for e in self.elements])
        self._dim = len(self.chemsys)
        self._description = description

    @property
    def entries(self) -> set[Entry]:
        return self._entries

    @property
    def elements(self) -> list[Element]:
        return self._elements

    @property
    def chemsys(self) -> str:
        return self._chemsys

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def description(self) -> NetworkEntryType:
        return self._description

    def as_dict(self) -> dict:
        """MSONable dict representation"""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "entries": list(self.entries),
            "description": self.description.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NetworkEntryType:
        """Load from MSONable dict"""
        return cls(
            MontyDecoder().process_decoded(d["entries"]),
            NetworkEntryType(d["description"]),
        )

    def __repr__(self) -> str:
        formulas = [entry.composition.reduced_formula for entry in self.entries]
        formulas.sort()
        return f"{self.description.name}: {','.join(formulas)}"

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.description == other.description:
                if self.chemsys == other.chemsys:
                    return self.entries == other.entries
        return False

    def __hash__(self):
        return hash((self.description, frozenset(self.entries)))


class DummyEntry(NetworkEntry):
    """
    A Dummy Entry that doesn't hold any info. This maybe useful for serving as an empty
    node to facilitate pathfinding to all nodes, etc.
    """

    def __init__(self):
        """Dummy node doesn't need any parameters"""
        self._entries = set()
        self._elements = []
        self._chemsys = ""
        self._dim = 0
        self._description = NetworkEntryType.Dummy

    def __repr__(self) -> str:
        return "Dummy Node"

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self):
        return hash("Dummy")
