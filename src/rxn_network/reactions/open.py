"""
A reaction class that builds reactions based on ComputedEntry objects under the
presence of an open entry (e.g. O2), and provides information about reaction
thermodynamics computed as changes in grand potential.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from pymatgen.core.composition import Element, Composition
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.reactions.computed import ComputedReaction


class OpenComputedReaction(ComputedReaction):
    """
    Extends the ComputedReaction class to add support for "open" reactions,
    where the reaction energy is calculated as a change in grand potential.
    """

    def __init__(
        self,
        entries: List[ComputedEntry],
        coefficients: Union[np.ndarray, List[float]],
        chempots: Dict[Element, float],
        data: Optional[Dict] = None,
        lowest_num_errors=None,
    ):
        """

        Args:
            entries: List of ComputedEntry objects.
            coefficients: List of reaction coefficients.
            chempots: Dict of chemical potentials corresponding to open elements
            data: Optional dict of data
            lowest_num_errors: number of "errors" encountered during reaction balancing
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
        reactant_entries: List[ComputedEntry],
        product_entries: List[ComputedEntry],
        chempots: Dict[Element, float] = None,
        data: Optional[Dict] = None,
    ):  # pylint: disable = W0221
        """

        Args:
            reactant_entries:
            product_entries:
            chempots:
            data:

        Returns:

        """

        reactant_comps = [e.composition.reduced_composition for e in reactant_entries]
        product_comps = [e.composition.reduced_composition for e in product_entries]
        coefficients, lowest_num_errors = cls._balance_coeffs(
            reactant_comps, product_comps
        )

        entries = list(reactant_entries) + list(product_entries)

        args = {
            "entries": entries,
            "coefficients": coefficients,
            "data": data,
            "lowest_num_errors": lowest_num_errors,
        }

        if not chempots:
            rxn = ComputedReaction(**args)  # type: ignore
        else:
            rxn = cls(chempots=chempots, **args)  # type: ignore

        return rxn

    @property
    def energy(self) -> float:
        """
        Returns (float):
            The calculated reaction energy.
        """
        calc_energies: Dict[Composition, float] = {}

        for entry in self.grand_entries:
            attr = "composition"
            if type(entry) == GrandPotPDEntry:
                attr = "original_comp"

            comp, factor = getattr(entry, attr).get_reduced_composition_and_factor()
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
    def elements(self) -> List[Element]:
        """
        List of elements in the reaction
        """
        return list(
            set(el for comp in self.compositions for el in comp.elements)
            - set(self.open_elems)
        )

    @property
    def total_chemical_system(self) -> str:
        """
        Chemical system string, including open elements
        """
        return "-".join(
            sorted([str(e) for e in set(self.elements) | set(self.open_elems)])
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

    def reverse(self):
        """

        Returns a copy of reaction with reactants/products swapped

        """
        return OpenComputedReaction(
            self.entries,
            -1 * self.coefficients,
            self.chempots,
            self.data,
            self.lowest_num_errors,
        )

    def __repr__(self):
        cp = f"({','.join([f'mu_{e}={m}' for e, m in self.chempots.items()])})"
        return f"{super().__repr__()} {cp}"
