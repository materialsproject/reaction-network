"""Class for intepolated entries"""

import numpy as np
from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from pymatgen.entries.computed_entries import ComputedEntry

from rxn_network.core.composition import Composition


class InterpolatedEntry(ComputedEntry):
    """
    Lightweight Entry object for computed data. Contains facilities
    for applying corrections to the energy attribute and for storing
    calculation parameters.
    """

    def __init__(
        self,
        composition: Composition,
        energy: float,
        correction: float = 0.0,
        energy_adjustments: list = None,
        parameters: dict = None,
        data: dict = None,
        entry_id: object = None,
    ):
        """
        Initializes an InterpolatedEntry.

        Args:
            composition (Composition): Composition of the entry. For
                flexibility, this can take the form of all the typical input
                taken by a Composition, including a {symbol: amt} dict,
                a string formula, and others.
            energy (float): Energy of the entry. Usually the final calculated
                energy from VASP or other electronic structure codes.
            correction (float): Manually set an energy correction, will ignore
                energy_adjustments if specified.
            energy_adjustments: An optional list of EnergyAdjustment to
                be applied to the energy. This is used to modify the energy for
                certain analyses. Defaults to None.
            parameters: An optional dict of parameters associated with
                the entry. Defaults to None.
            data: An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        super().__init__(
            composition,
            energy,
            correction=correction,
            energy_adjustments=energy_adjustments,
            parameters=parameters,
            data=data,
            entry_id=entry_id,
        )

    def to_grand_entry(self, chempots):
        """
        Convert a GibbsComputedEntry to a GrandComputedEntry.

        Args:
            chempots: A dictionary of {element: chempot} pairs.

        Returns:
            A GrandComputedEntry.
        """
        return GrandPotPDEntry(self, chempots)

    @property
    def is_experimental(self) -> bool:
        """Returns True by default."""
        return False

    def __repr__(self):
        output = [
            (
                f"InterpolatedEntry | {self.composition.formula} "
                f"({self.composition.reduced_formula})"
            ),
            f"Energy  = {self.energy:.4f}",
        ]
        return "\n".join(output)

    def __eq__(self, other):
        if not type(other) is type(self):
            return False

        if not np.isclose(self.energy, other.energy):
            return False

        if getattr(self, "entry_id", None) and getattr(other, "entry_id", None):
            return self.entry_id == other.entry_id

        if self.composition != other.composition:
            return False

        return True

    def __hash__(self):
        return hash(
            (
                self.composition,
                self.energy,
                self.entry_id,
            )
        )
