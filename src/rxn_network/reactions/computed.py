" A ComputedReaction class to generate from ComputedEntry objects "
from typing import List, Dict

import numpy as np
from pymatgen import Element
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
        self.reactant_entries = list(reactant_entries)
        self.product_entries = list(product_entries)
        self.coefficients = coefficients
        self.lowest_num_errors = kwargs.get("lowest_num_errors", None)

        all_entries = self.reactant_entries + self.product_entries

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


class OpenComputedReaction(ComputedReaction):
    """
    Extends the ComputedReaction class to add support for "open" reactions,
    where the reaction energy is calculated as a change in grand potential.
    """
    def __init__(
        self,
        reactant_entries: List[Entry],
        product_entries: List[Entry],
        coefficients: np.array,
        chempots,
        **kwargs
    ):
        """
        Args:
            reactant_entries ([ComputedEntry]): List of reactant_entries.
            product_entries ([ComputedEntry]): List of product_entries.
        """
        super().__init__(reactant_entries, product_entries, coefficients, **kwargs)

        self.chempots = chempots
        self.open_elems = list(chempots.keys())

        self.grand_compositions = []
        self.grand_coefficients = []

        self.reactant_grand_entries = []
        for e, coeff in zip(reactant_entries, self.reactant_coeffs.values()):
            comp = e.composition.reduced_composition
            if len(comp.elements) == 1 and comp.elements[0] in \
                    self.open_elems:
                continue
            self.reactant_grand_entries.append(GrandPotPDEntry(e, chempots))
            self.grand_coefficients.append(coeff)
            self.grand_compositions.append(comp)

        self.product_grand_entries = []
        for e, coeff in zip(product_entries, self.product_coeffs.values()):
            comp = e.composition.reduced_composition
            if len(comp.elements) == 1 and comp.elements[0] in \
                    self.open_elems:
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
        return list(set(el for comp in self.compositions for el in comp.elements) -
                    set(self.open_elems))

    @property
    def energy_per_atom(self) -> float:
        """
        Returns (float):
            The calculated reaction energy in eV, divided by the total number of
            atoms in the reaction.
        """
        return self.energy / self.num_atoms


    @classmethod
    def balance(
        cls, reactant_entries: List[Entry], product_entries: List[Entry], chempots
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
        coefficients, lowest_num_errors = cls._balance_coeffs(reactant_comps,
                                                           product_comps)

        return cls(
            reactant_entries=reactant_entries,
            product_entries=product_entries,
            coefficients=coefficients,
            chempots=chempots,
            lowest_num_errors=lowest_num_errors
        )