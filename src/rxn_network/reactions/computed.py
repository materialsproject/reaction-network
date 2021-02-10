" A ComputedReaction class to generate from ComputedEntry objects "
from typing import List

import numpy as np
from pymatgen.entries import Entry
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
        reactant_entries: List[Entry],
        product_entries: List[Entry],
        coefficients: np.array,
        **kwargs
    ):
        """
        Args:
            reactant_entries ([ComputedEntry]): List of reactant_entries.
            product_entries ([ComputedEntry]): List of product_entries.
        """
        self.reactant_entries = reactant_entries
        self.product_entries = product_entries
        self.coefficients = coefficients

        self.lowest_num_errors = kwargs.get("lowest_num_errors", None)

        all_entries = reactant_entries + product_entries

        reactant_coeffs = None
        product_coeffs = None

        if coefficients.any():
            reactant_coeffs = {
                e.composition.reduced_composition: c
                for e, c in zip(all_entries, coefficients)
                if c < 0
            }
            product_coeffs = {
                e.composition.reduced_composition: c
                for e, c in zip(all_entries, coefficients)
                if c > 0
            }

        super().__init__(reactant_coeffs, product_coeffs)

    @property
    def entries(self):
        """
        Equivalent of all_comp but returns entries, in the same order as the
        coefficients.

        #Matt: do you needed it ordered by coefficients? all_comp was not ordered as such
        """
        return self.reactant_entries + self.product_entries

    @property
    def energy(self) -> float:
        """
        Returns (float):
            The calculated reaction energy.
        """
        calc_energies = {}

        for entry in self.reactant_entries + self.product_entries:
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
    def energy_uncertainty(self):
        """
        Calculates the uncertainty in the reaction energy based on the uncertainty in the
        energies of the products and reactants
        """

        calc_energies = {}

        for entry in self.reactant_entries + self.product_entries:
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

    @classmethod
    def balance(
        cls, reactant_entries: List[Entry], product_entries: List[Entry]
    ):  # pylint: disable = W0221
        """
        Balances and returns a new ComputedReaction

        Reactants and products to be specified as list of
        pymatgen.core.structure.Composition.  e.g., [comp1, comp2]

        Args:
            reactants ([Composition]): List of reactants.
            products ([Composition]): List of products.
        """

        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors = cls._balance_coeffs(reactant_comps,
                                                           product_comps)

        return cls(
            reactant_entries=reactant_entries,
            product_entries=product_entries,
            coefficients=coefficients,
            lowest_num_errors=lowest_num_errors
        )
