"""
Entry objects used in a Network. This holds multiple entries and can be used as data
for a graph node.
"""
from enum import Enum, auto
from typing import List

from monty.json import MSONable
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
            entries: list of ComputedEntry-like objects
            description: Node type
        """
        self.entries = set(entries)
        self.elements = sorted(
            list({elem for entry in entries for elem in entry.composition.elements})
        )
        self.chemsys = "-".join([str(e) for e in self.elements])
        self.dim = len(self.chemsys)
        self.description = description

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
    """A Dummy Entry that doesn't hold any info"""

    def __init__(self):  # pylint: disable=W0231
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
