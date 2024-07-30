"""A reaction class that builds reactions based on ComputedEntry objects and provides
information about reaction thermodynamics.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict

import numpy as np
from uncertainties import ufloat

from rxn_network.reactions.basic import BasicReaction

if TYPE_CHECKING:
    from pymatgen.core.composition import Element
    from pymatgen.entries.computed_entries import ComputedEntry

    from rxn_network.core import Composition


class ComputedReaction(BasicReaction):
    """Convenience class to generate a reaction from ComputedEntry objects, with
    some additional attributes, such as a reaction energy based on computed
    energies. This class also balances the reaction.
    """

    def __init__(
        self,
        entries: list[ComputedEntry],
        coefficients: np.ndarray | list[float],
        data: dict | None = None,
        lowest_num_errors: int = 0,
    ):
        """
        Args:
            entries: List of ComputedEntry objects.
            coefficients: List or array of reaction coefficients.
            data: Optional dict of data
            lowest_num_errors: number of "errors" encountered during reaction balancing.
        """
        self._entries = list(entries)
        self.reactant_entries = [entry for entry, coeff in zip(entries, coefficients) if coeff < 0]
        self.product_entries = [entry for entry, coeff in zip(entries, coefficients) if coeff > 0]
        compositions = [e.composition.reduced_composition for e in entries]

        super().__init__(compositions, coefficients, data=data, lowest_num_errors=lowest_num_errors)

    @classmethod
    def balance(
        cls,
        reactant_entries: list[ComputedEntry],
        product_entries: list[ComputedEntry],
        data: dict | None = None,
    ) -> ComputedReaction:
        """Balances and returns a new ComputedReaction.

        Reactants and products to be specified as a collection (list, set, etc.) of
        ComputedEntry objects.

        Args:
            reactant_entries: List of reactant entries
            product_entries: List of product entries
            data: Optional dict of data
        """
        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors, num_constraints = cls._balance_coeffs(reactant_comps, product_comps)

        if not data:
            data = {}
        data["num_constraints"] = num_constraints

        return cls(
            entries=list(reactant_entries) + list(product_entries),
            coefficients=coefficients,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )

    def get_new_temperature(self, new_temperature: float, entries_at_temp: Dict[Composition, ComputedEntry] = None) -> ComputedReaction:
        """Returns a new reaction with the temperature changed.

        Args:
            new_temperature: New temperature in Kelvin
        """
        try:
            if entries_at_temp is None:
                new_entries = [e.get_new_temperature(new_temperature) for e in self.entries]
            else:
                new_entries = [entries_at_temp.get(e.composition) for e in self.entries]
        except AttributeError as e:
            raise AttributeError(
                "One or more of the entries in the reaction is not associated with a"
                " temperature. Please use the GibbsComputedEntry class for all entries"
                " in the reaction."
            ) from e

        return ComputedReaction(
            new_entries,
            self.coefficients,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    @cached_property
    def energy(self) -> float:
        """Returns (float):
        The calculated reaction energy.
        """
        calc_energies: dict[Composition, float] = {}

        for entry in self._entries:
            (comp, factor) = entry.composition.get_reduced_composition_and_factor()
            calc_energies[comp] = min(calc_energies.get(comp, float("inf")), entry.energy / factor)

        return sum(amt * calc_energies[c] for amt, c in zip(self.coefficients, self.compositions))

    @cached_property
    def energy_per_atom(self) -> float:
        """Returns (float):
        The calculated reaction energy in eV, divided by the total number of atoms
        in the reaction. Cached for speedup.
        """
        return self.energy / self.num_atoms

    @cached_property
    def energy_uncertainty(self) -> float:
        """Calculates the uncertainty in the reaction energy based on the uncertainty in
        the energies of the reactants/products. Cached for speedup.
        """
        calc_energies: dict[Composition, ufloat] = {}

        for entry in self._entries:
            (comp, factor) = entry.composition.get_reduced_composition_and_factor()
            energy_ufloat = ufloat(entry.energy, entry.correction_uncertainty)
            calc_energies[comp] = min(calc_energies.get(comp, float("inf")), energy_ufloat / factor)

        energy_with_uncertainty = sum(amt * calc_energies[c] for amt, c in zip(self.coefficients, self.compositions))

        return energy_with_uncertainty.s  # type: ignore

    @cached_property
    def energy_uncertainty_per_atom(self) -> float:
        """Returns the energy_uncertainty divided by the total number of atoms in the
        reaction.
        """
        return self.energy_uncertainty / self.num_atoms

    @property
    def entries(self) -> list[ComputedEntry]:
        """Returns a copy of the entries."""
        return self._entries

    def copy(self) -> ComputedReaction:
        """Returns a copy of the Reaction object."""
        return ComputedReaction(self.entries, self.coefficients, self.data, self.lowest_num_errors)

    def reverse(self) -> ComputedReaction:
        """Returns a reversed reaction (i.e. sides flipped)."""
        return ComputedReaction(self.entries, -1 * self.coefficients, self.data, self.lowest_num_errors)

    def normalize_to(self, comp: Composition, factor: float = 1) -> ComputedReaction:
        """Normalizes the reaction to one of the compositions via the provided factor.

        By default, normalizes such that the composition given has a coefficient of
        1.

        Args:
            comp: Composition object to normalize to
            factor: factor to normalize to. Defaults to 1.
        """
        coeffs = self.coefficients.copy()
        scale_factor = abs(1 / coeffs[self.compositions.index(comp)] * factor)
        coeffs *= scale_factor
        return ComputedReaction(self.entries, coeffs, self.data, self.lowest_num_errors)

    def normalize_to_element(self, element: Element, factor: float = 1) -> ComputedReaction:
        """Normalizes the reaction to one of the elements.
        By default, normalizes such that the amount of the element is 1.
        Another factor can be specified.

        Args:
            element (Element/Species): Element to normalize to.
            factor (float): Factor to normalize to. Defaults to 1.
        """
        all_comp = self.compositions
        coeffs = self.coefficients.copy()
        current_el_amount = sum(all_comp[i][element] * abs(coeffs[i]) for i in range(len(all_comp))) / 2
        scale_factor = factor / current_el_amount
        coeffs *= scale_factor
        return ComputedReaction(self.entries, coeffs, self.data, self.lowest_num_errors)

    def get_entry_idx_vector(self, n: int) -> np.ndarray:
        """Maps coefficients to a vector of length n on corresponding entry indices.

        Args:
            n: Total number of entries (determines length of vector)

        Returns:
            Vector containing reaction coefficients on the corresponding entry indices.
        """
        indices = [e.data["idx"] for e in self.entries]

        v = np.zeros(n)
        v[indices] = self.coefficients
        return v

    def __eq__(self, other) -> bool:
        is_equal = BasicReaction.__eq__(self, other)

        if is_equal:
            is_equal = np.isclose(  # type: ignore
                self.energy_per_atom, other.energy_per_atom
            )

        return is_equal

    def __hash__(self):
        """Must be redefined due to overriding __eq__."""
        return hash(
            (self.chemical_system, tuple(sorted(self.coefficients)))
        )  # not checking here for reactions that are multiples (too expensive)
