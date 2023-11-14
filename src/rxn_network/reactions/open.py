"""A reaction class that builds reactions based on ComputedEntry objects under the presence
of an open entry (e.g. O2), and provides information about reaction thermodynamics
computed as changes in grand potential.
"""
from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from pymatgen.core.composition import Element

from rxn_network.reactions.computed import ComputedReaction

if TYPE_CHECKING:
    import numpy as np
    from pymatgen.entries.computed_entries import ComputedEntry

    from rxn_network.core import Composition


class OpenComputedReaction(ComputedReaction):
    """Extends the ComputedReaction class to add support for "open" reactions,
    where the reaction energy is calculated as a change in grand potential.
    """

    def __init__(
        self,
        entries: list[ComputedEntry],
        coefficients: np.ndarray | list[float],
        chempots: dict[Element, float],
        data: dict | None = None,
        lowest_num_errors: int = 0,
    ):
        """
        Args:
            entries: List of ComputedEntry objects.
            coefficients: List of reaction coefficients.
            chempots: Dict of chemical potentials corresponding to open elements.
            data: Optional dict of data.
            lowest_num_errors: number of "errors" encountered during reaction
                balancing.
        """
        super().__init__(
            entries=entries,
            coefficients=coefficients,
            data=data,
            lowest_num_errors=lowest_num_errors,
        )

        self.chempots = chempots
        self.open_elems = list(chempots.keys())

        grand_entries = []
        for e in entries:
            comp = e.composition.reduced_composition
            if len(comp.elements) == 1 and comp.elements[0] in self.open_elems:
                grand_entries.append(e)
            else:
                grand_entries.append(GrandPotPDEntry(e, chempots))

        self.grand_entries = grand_entries

    @classmethod
    def balance(  # type: ignore
        cls,
        reactant_entries: list[ComputedEntry],
        product_entries: list[ComputedEntry],
        chempots: dict[Element, float],
        data: dict | None = None,
    ) -> OpenComputedReaction:
        """Balances and returns a new ComputedReaction.

        Reactants and products to be specified as a collection (list, set, etc.) of
        ComputedEntry objects.

        A dictionary of open elements and their corresponding chemical potentials must
        be supplied.

        Args:
            reactant_entries: List of reactant entries
            product_entries: List of product entries
            chempots: Dict of chemical potentials corresponding to open
                element(s)
            data: Optional dict of data
        """
        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors, num_constraints = cls._balance_coeffs(reactant_comps, product_comps)

        if not data:
            data = {}
        data["num_constraints"] = num_constraints

        entries = list(reactant_entries) + list(product_entries)

        kwargs = {
            "entries": entries,
            "coefficients": coefficients,
            "data": data,
            "lowest_num_errors": lowest_num_errors,
        }

        return ComputedReaction(**kwargs) if not chempots else cls(chempots=chempots, **kwargs)

    def get_new_temperature(self, new_temperature: float) -> OpenComputedReaction:
        """Returns a new reaction with the temperature changed.

        Args:
            new_temperature: New temperature in Kelvin
        """
        try:
            new_entries = [e.get_new_temperature(new_temperature) for e in self.entries]
        except AttributeError as e:
            raise AttributeError(
                "One or more of the entries in the reaction is not associated with a"
                " temperature. Please use the GibbsComputedEntry class for all entries"
                " in the reaction."
            ) from e

        return OpenComputedReaction(
            new_entries,
            self.coefficients,
            chempots=self.chempots,
            data=self.data,
            lowest_num_errors=self.lowest_num_errors,
        )

    @property
    def energy(self) -> float:
        """Returns (float):
        The calculated reaction energy.
        """
        calc_energies: dict[Composition, float] = {}

        for entry in self.grand_entries:
            attr = "composition"
            if isinstance(entry, GrandPotPDEntry):
                attr = "original_comp"

            comp, factor = getattr(entry, attr).get_reduced_composition_and_factor()
            calc_energies[comp] = min(calc_energies.get(comp, float("inf")), entry.energy / factor)

        return sum(amt * calc_energies[c] for amt, c in zip(self.coefficients, self.compositions))

    @property
    def elements(self) -> list[Element]:
        """List of elements in the reaction."""
        return list({el for comp in self.compositions for el in comp.elements} - set(self.open_elems))

    @property
    def total_chemical_system(self) -> str:
        """Chemical system string, including open elements."""
        return "-".join(sorted([str(e) for e in set(self.elements) | set(self.open_elems)]))

    def copy(self) -> OpenComputedReaction:
        """Returns a copy of the OpenComputedReaction object."""
        return OpenComputedReaction(
            self.entries,
            self.coefficients,
            self.chempots,
            self.data,
            self.lowest_num_errors,
        )

    def reverse(self):
        """Returns a copy of reaction with reactants/products swapped."""
        return OpenComputedReaction(
            self.entries,
            -1 * self.coefficients,
            self.chempots,
            self.data,
            self.lowest_num_errors,
        )

    @cached_property
    def reactant_atomic_fractions(self) -> dict:
        """Returns the atomic mixing ratio of reactants in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        return {
            c.reduced_composition: -coeff * sum(c[el] for el in self.elements) / self.num_atoms
            for c, coeff in self.reactant_coeffs.items()
        }

    @cached_property
    def product_atomic_fractions(self) -> dict:
        """Returns the atomic mixing ratio of reactants in the reaction."""
        if not self.balanced:
            raise ValueError("Reaction is not balanced")

        return {
            c.reduced_composition: sum(c[el] for el in self.elements) / self.num_atoms
            for c, coeff in self.product_coeffs.items()
        }

    @classmethod
    def from_computed_rxn(cls, reaction: ComputedReaction, chempots: dict[Element, float]) -> OpenComputedReaction:
        """Generate an OpenComputedReaction from a ComputedReaction object and chemical potential dict.

        Args:
            reaction: ComputedReaction object
            chempots: Dict of chemical potentials corresponding to open element(s)

        Returns:
            OpenComputedReaction object
        """
        return cls(
            entries=reaction.entries.copy(),
            coefficients=reaction.coefficients.copy(),
            chempots=chempots,
            data=reaction.data.copy(),
            lowest_num_errors=reaction.lowest_num_errors,
        )

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the reaction."""
        d = super().as_dict()
        d["chempots"] = {el.symbol: u for el, u in self.chempots.items()}
        return d

    @classmethod
    def from_dict(cls, d) -> OpenComputedReaction:
        """Returns an OpenComputedReaction object from a dictionary representation."""
        d["chempots"] = {Element(symbol): u for symbol, u in d["chempots"].items()}
        return super().from_dict(d)

    def __repr__(self) -> str:
        cp = f"({','.join([f'mu_{e}={m}' for e, m in self.chempots.items()])})"
        return f"{super().__repr__()} {cp}"
