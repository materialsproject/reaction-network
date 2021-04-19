from typing import List, Dict, Optional

import numpy as np

from pymatgen.core.composition import Element
from pymatgen.entries import Entry
from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from rxn_network.reactions.computed import ComputedReaction


class OpenComputedReaction(ComputedReaction):
    """
    Extends the ComputedReaction class to add support for "open" reactions,
    where the reaction energy is calculated as a change in grand potential.
    """

    def __init__(
        self,
        entries: List[Entry],
        coefficients: np.array,
        chempots,
        data: Optional[Dict] = None,
        lowest_num_errors=None,
    ):
        """
        Args:
            reactant_entries ([ComputedEntry]): List of reactant_entries.
            product_entries ([ComputedEntry]): List of product_entries.
        """
        super().__init__(entries, coefficients, data, lowest_num_errors)

        self.chempots = chempots
        self.open_elems = list(chempots.keys())

        self.grand_compositions = []
        self.grand_coefficients = []

        self.reactant_grand_entries = []
        for e, coeff in zip(self.reactant_entries, self.reactant_coeffs.values()):
            comp = e.composition.reduced_composition
            if len(comp.elements) == 1 and comp.elements[0] in self.open_elems:
                continue
            self.reactant_grand_entries.append(GrandPotPDEntry(e, chempots))
            self.grand_coefficients.append(coeff)
            self.grand_compositions.append(comp)

        self.product_grand_entries = []
        for e, coeff in zip(self.product_entries, self.product_coeffs.values()):
            comp = e.composition.reduced_composition
            if len(comp.elements) == 1 and comp.elements[0] in self.open_elems:
                continue
            self.product_grand_entries.append(GrandPotPDEntry(e, chempots))
            self.grand_coefficients.append(coeff)
            self.grand_compositions.append(comp)

    @property
    def energy(self) -> float:
        """
        Returns (float):
            The calculated reaction energy.
        """
        calc_energies = {}

        for entry in self.reactant_grand_entries + self.product_grand_entries:
            (comp, factor) = entry.original_comp.get_reduced_composition_and_factor()
            calc_energies[comp] = min(
                calc_energies.get(comp, float("inf")), entry.energy / factor
            )

        return sum(
            [
                amt * calc_energies[c]
                for amt, c in zip(self.grand_coefficients, self.grand_compositions)
            ]
        )

    @property
    def elements(self) -> List[Element]:
        """
        List of elements in the reaction
        """
        return list(
            set(el for comp in self.compositions for el in comp.elements)
            - set(self.open_elems)
        )

    def copy(self) -> "OpenComputedReaction":
        """
        Returns a copy of the OpenComputedReaction object.
        """
        return OpenComputedReaction(
            self.entries,
            self.coefficients,
            self.chempots,
            self.data,
            self.lowest_num_errors,
        )

    @classmethod
    def balance(
        cls,
        reactant_entries: List[Entry],
        product_entries: List[Entry],
        chempots,
        data=None,
    ):  # pylint: disable = W0221
        """
        Balances and returns a new OpenComputedReaction

        Reactants and products to be specified as list of
        pymatgen.core.structure.Composition.  e.g., [comp1, comp2]

        Args:
            reactants ([Composition]): List of reactants.
            products ([Composition]): List of products.
        """
        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors = cls._balance_coeffs(
            reactant_comps, product_comps
        )

        if not coefficients.any():
            coefficients = []

        return cls(
            entries=list(reactant_entries) + list(product_entries),
            coefficients=list(coefficients),
            chempots=chempots,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )
