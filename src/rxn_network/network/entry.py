from typing import List
from enum import Enum, auto

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries import Entry
from scipy.interpolate import interp1d


class NetworkEntryType(Enum):
    " Describes the Network Entry Type "
    Reactants = auto()
    Products = auto()
    Starter = auto()
    Target = auto()
    Dummy = auto()


class NetworkEntry(MSONable):
    """
    Helper class for describing combinations of ComputedEntry-like objects in context
    of a reaction network. Necessary for implementation in NetworkX (and useful
    for other network packages!)
    """

    def __init__(self, entries: List[Entry], description: NetworkEntryType):
        """
        Args:
            entries [ComputedEntry]: list of ComputedEntry-like objects
            description (str): Node type, as selected from:
                "R" (reactants), "P" (products),
                "S" (starters/precursors), "T" (target),
                "D" (dummy)
        """
        self.entries = set(entries)
        self.chemsys = "-".join(
            sorted(
                {str(el) for entry in self.entries for el in entry.composition.elements}
            )
        )

        self.description = description

    def __repr__(self):
        formulas = [entry.composition.reduced_formula for entry in self.entries]
        formulas.sort()
        return f"{self.description}: {','.join(formulas)}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.description == other.description:
                if self.chemsys == other.chemsys:
                    return self.entries == other.entries
        else:
            return False

    def __hash__(self):
        return hash((self.description, frozenset(self.entries)))


class DummyEntry(NetworkEntry):
    " A Dummy Entry that doesn't hold any info "

    def __init__(self):
        " Dummy node doesn't need any parameters "
        pass

    @property
    def entries(self):
        return []

    @property
    def chemsys(self):
        return ""

    @property
    def description(self):
        return NetworkEntryType.Dummy

    def __repr__(self):
        return "Dummy Node"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash("Dummy")