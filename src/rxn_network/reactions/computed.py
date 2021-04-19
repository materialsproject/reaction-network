" A ComputedReaction class to generate from ComputedEntry objects "
from typing import List, Dict, Optional

import numpy as np
from pymatgen.core.composition import Element
from pymatgen.entries import Entry
from pymatgen.analysis.phase_diagram import PDEntry, GrandPotPDEntry
from uncertainties import ufloat

from rxn_network.reactions.basic import BasicReaction


class ComputedReaction(BasicReaction):
    """
    Convenience class to generate a reaction from ComputedEntry objects, with
    some additional attributes, such as a reaction energy based on computed
    energies. Will balance the reaction.
    """

    def __init__(
        self,
        entries: List[Entry],
        coefficients: List[float],
        data: Optional[Dict] = None,
        lowest_num_errors: Optional[int] = None,
    ):
        """
        Args:
            entries([ComputedEntry]): List of ComputedEntry objects.
            coefficients([float]): List of reaction coefficients.
        """
        self._entries = list(entries)
        self.reactant_entries = [
            entry for entry, coeff in zip(entries, coefficients) if coeff < 0
        ]
        self.product_entries = [
            entry for entry, coeff in zip(entries, coefficients) if coeff > 0
        ]
        compositions = [e.composition.reduced_composition for e in entries]

        super().__init__(
            compositions, coefficients, data=data, lowest_num_errors=lowest_num_errors
        )

    @property
    def entries(self):
        """
        Equivalent of all_comp but returns entries, in the same order as the
        coefficients.

        """
        return self._entries

    @property
    def energy(self) -> float:
        """
        Returns (float):
            The calculated reaction energy.
        """
        calc_energies = {}

        for entry in self._entries:
            (comp, factor) = entry.composition.get_reduced_composition_and_factor()
            calc_energies[comp] = min(
                calc_energies.get(comp, float("inf")), entry.energy / factor
            )

        return sum(
            [
                amt * calc_energies[c]
                for amt, c in zip(self.coefficients, self.compositions)
            ]
        )

    @property
    def energy_per_atom(self) -> float:
        """
        Returns (float):
            The calculated reaction energy in eV, divided by the total number of
            atoms in the reaction.
        """
        return self.energy / self.num_atoms

    @property
    def energy_uncertainty(self):
        """
        Calculates the uncertainty in the reaction energy based on the uncertainty in the
        energies of the products and reactants
        """

        calc_energies = {}

        for entry in self._entries:
            (comp, factor) = entry.composition.get_reduced_composition_and_factor()
            energy_ufloat = ufloat(entry.energy, entry.correction_uncertainty)
            calc_energies[comp] = min(
                calc_energies.get(comp, float("inf")), energy_ufloat / factor
            )

        return sum(
            [
                amt * calc_energies[c]
                for amt, c in zip(self.coefficients, self.compositions)
            ]
        )

    def copy(self) -> "ComputedReaction":
        """
        Returns a copy of the Reaction object.
        """
        return ComputedReaction(
            self.entries, self.coefficients, self.data, self.lowest_num_errors
        )

    @property
    def energy_uncertainty_per_atom(self):
        return self.energy_uncertainty / self.num_atoms

    @classmethod
    def balance(
        cls,
        reactant_entries: List[Entry],
        product_entries: List[Entry],
        data: Optional[Dict] = None,
    ):  # pylint: disable = W0221
        """
        Balances and returns a new ComputedReaction.

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
            coefficients=coefficients,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )
