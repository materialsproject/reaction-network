import hashlib
import json
import os
from itertools import chain, combinations

import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry, GibbsComputedStructureEntry
from scipy.interpolate import interp1d

from rxn_network.data import G_COMPOUNDS, G_ELEMS, G_GASES
from typing import List, Optional


def _new_pdentry_hash(self):  # necessary fix, will be updated in pymatgen in future
    data_md5 = hashlib.md5(
        f"{self.composition.formula}_" f"{self.energy}".encode("utf-8")
    ).hexdigest()
    return int(data_md5, 16)

def _new_gibbsentry_hash(self):
    data_md5 = hashlib.md5(
        f"{self.composition}_"
        f"{self.formation_enthalpy_per_atom}_{self.entry_id}_"
        f"{self.temp}".encode("utf-8")
    ).hexdigest()
    return int(data_md5, 16)

PDEntry.__hash__ = _new_pdentry_hash
GibbsComputedStructureEntry.__hash__ = _new_gibbsentry_hash


class CustomEntry(PDEntry):
    def __init__(self, composition, energy_dict, temp=None, name=None, attribute=None):
        composition = Composition(composition)

        if not temp:
            temp = 300

        super().__init__(
            composition, energy_dict[str(temp)], name=name, attribute=attribute
        )
        self.temp = temp
        self.energy_dict = energy_dict

    def set_temp(self, temp):
        super().__init__(
            self.composition,
            self.energy_dict[str(temp)],
            name=self.name,
            attribute=self.attribute,
        )

    def __repr__(self):
        return super().__repr__() + f" (T={self.temp} K)"


class RxnEntries(MSONable):
    """
    Helper class for describing combinations of ComputedEntry-like objects in context
    of a reaction network. Necessary for implementation in NetworkX (and useful
    for other network packages!)
    """

    def __init__(self, entries, description):
        """
        Args:
            entries [ComputedEntry]: list of ComputedEntry-like objects
            description (str): Node type, as selected from:
                "R" (reactants), "P" (products),
                "S" (starters/precursors), "T" (target),
                "D" (dummy)
        """
        self._entries = set(entries) if entries else None
        self._chemsys = (
            "-".join(
                sorted(
                    {
                        str(el)
                        for entry in self._entries
                        for el in entry.composition.elements
                    }
                )
            )
            if entries
            else None
        )

        if description in ["r", "R", "reactants", "Reactants"]:
            self._description = "R"
        elif description in ["p", "P", "products", "Products"]:
            self._description = "P"
        elif description in [
            "s",
            "S",
            "precursors",
            "Precursors",
            "starters",
            "Starters",
        ]:
            self._description = "S"
        elif description in ["t", "T", "target", "Target"]:
            self._description = "T"
        elif description in ["d", "D", "dummy", "Dummy"]:
            self._description = "D"
        else:
            self._description = description

    @property
    def entries(self):
        return self._entries

    @property
    def description(self):
        return self._description

    @property
    def chemsys(self):
        return self._chemsys

    def __repr__(self):
        if self._description == "D":
            return "Dummy Node"

        formulas = [entry.composition.reduced_formula for entry in self._entries]
        formulas.sort()
        if not self._description:
            return f"{','.join(formulas)}"
        else:
            return f"{self._description}: {','.join(formulas)}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.description == other.description:
                if self.chemsys == other.chemsys:
                    return self.entries == other.entries
        else:
            return False

    def __hash__(self):
        if not self._description or self._description == "D":
            return hash(self._description)
        else:
            return hash((self._description, frozenset(self._entries)))