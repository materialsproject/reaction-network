"""
A reaction class that builds reactions based on ComputedEntry objects and provides
information about reaction thermodynamics.
"""
from typing import Dict, List, Optional, Union

import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from uncertainties import ufloat

from rxn_network.reactions.basic import BasicReaction


class ComputedReaction(BasicReaction):
    """
    Convenience class to generate a reaction from ComputedEntry objects, with
    some additional attributes, such as a reaction energy based on computed
    energies. This class also balances the reaction.
    """

    def __init__(
        self,
        entries: List[ComputedEntry],
        coefficients: Union[np.ndarray, List[float]],
        data: Optional[Dict] = None,
        lowest_num_errors: Union[int, float] = 0,
    ):
        """
        Args:
            entries: List of ComputedEntry objects.
            coefficients: List of reaction coefficients.
            data: Optional dict of data
            lowest_num_errors: number of "errors" encountered during reaction balancing
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

    @classmethod
    def balance(
        cls,
        reactant_entries: List[ComputedEntry],
        product_entries: List[ComputedEntry],
        data: Optional[Dict] = None,
    ):  # pylint: disable = W0221
        """
        Balances and returns a new ComputedReaction.

        Reactants and products to be specified as list of
        pymatgen.core.structure.Composition.  e.g., [comp1, comp2]

        Args:
            reactant_entries: List of reactant entries
            product_entries: List of product entries
            data: Optional dict of data
        """
        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors = cls._balance_coeffs(
            reactant_comps, product_comps
        )

        return cls(
            entries=list(reactant_entries) + list(product_entries),
            coefficients=coefficients,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )

    @property
    def energy(self) -> float:
        """
        Returns (float):
            The calculated reaction energy.
        """
        calc_energies: Dict[Composition, float] = {}

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
        energies of the reactants/products
        """

        calc_energies = {}

        for entry in self._entries:
            (comp, factor) = entry.composition.get_reduced_composition_and_factor()
            energy_ufloat = ufloat(entry.energy, entry.correction_uncertainty)
            calc_energies[comp] = min(
                calc_energies.get(comp, float("inf")), energy_ufloat / factor
            )

        energy_with_uncertainty = sum(
            [
                amt * calc_energies[c]
                for amt, c in zip(self.coefficients, self.compositions)
            ]
        )

        return energy_with_uncertainty.s

    @property
    def energy_uncertainty_per_atom(self):
        return self.energy_uncertainty / self.num_atoms

    @property
    def entries(self):
        """
        Returns a copy of the entries
        """
        return self._entries

    def copy(self) -> "ComputedReaction":
        """
        Returns a copy of the Reaction object
        """
        return ComputedReaction(
            self.entries, self.coefficients, self.data, self.lowest_num_errors
        )

    def reverse(self):
        """
        Returns a reversed reaction (i.e. sides flipped)

        """
        return ComputedReaction(
            self.entries, -1 * self.coefficients, self.data, self.lowest_num_errors
        )

    def __hash__(self):
        return BasicReaction.__hash__(self)

    def __eq__(self, other):
        eq = BasicReaction.__eq__(self, other)
        if not eq:
            return False
        else:
            return np.isclose(self.energy_per_atom, other.energy_per_atom)
