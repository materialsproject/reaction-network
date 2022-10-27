"""
Entry objects used in a Network. These network entry objects hold multiple entries and
can be used as data for a node in the graph.
"""
from enum import Enum, auto
from typing import List

from monty.json import MSONable
from monty.serialization import MontyDecoder
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
    of a reaction network.
    """

    def __init__(self, entries: List[Entry], description: NetworkEntryType):
        """
        Args:
            entries: list of Entry-like objects
            description: Node type (e.g., Precursors, Target... see NetworkEntryType
                class)
        """
        self.entries = set(entries)
        self.elements = sorted(
            list({elem for entry in entries for elem in entry.composition.elements})
        )
        self.chemsys = "-".join([str(e) for e in self.elements])
        self.dim = len(self.chemsys)
        self.description = description

    def as_dict(self):
        """MSONable dict representation"""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "entries": list(self.entries),
            "description": self.description.value,
        }

    @classmethod
    def from_dict(cls, d):
        """Load from MSONable dict"""
        return cls(
            MontyDecoder().process_decoded(d["entries"]),
            NetworkEntryType(d["description"]),
        )

    def __repr__(self):
        formulas = [entry.composition.reduced_formula for entry in self.entries]
        formulas.sort()
        return f"{self.description.name}: {','.join(formulas)}"

    def __eq__(self, other):
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

    def __init__(self):  # pylint: disable=super-init-not-called
        """Dummy node doesn't need any parameters"""

    @property
    def entries(self):
        """No entries in DummyEntry"""
        return []

    @property
    def chemsys(self):
        """No Chemsys to DummyEntry"""
        return ""

    @property
    def description(self):
        """DummyEntry is always of type Dummy"""
        return NetworkEntryType.Dummy

    def __repr__(self):
        return "Dummy Node"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash("Dummy")
