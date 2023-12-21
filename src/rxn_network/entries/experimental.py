"""Implements an Entry that looks up NIST pre-tabulated Gibbs free energies."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from pymatgen.entries.computed_entries import ComputedEntry, EnergyAdjustment
from scipy.interpolate import interp1d

from rxn_network.core import Composition
from rxn_network.utils.funcs import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:
    from pymatgen.core.periodic_table import Element


class ExperimentalReferenceEntry(ComputedEntry):
    """An Entry class for experimental reference data, to be sub-classed for specific data
    sources.  Given a composition, automatically finds the Gibbs free energy of
    formation, dGf(T) from tabulated reference values.
    """

    REFERENCES: dict = {}
    DEPRECATED: list = []

    def __init__(
        self,
        composition: Composition,
        temperature: float,
        energy_adjustments: list[EnergyAdjustment] | None = None,
        data: dict | None = None,
    ):
        """
        Args:
            composition: Composition object
            temperature: Temperature in Kelvin. If temperature is not selected within
                the range of the reference  data (see self._validate_temperature), then
                this will raise an error.
            energy_adjustments: A list of EnergyAdjustments to apply to the entry.
            data: Optional dictionary containing entry data.
        """
        self.is_deprecated = composition.reduced_formula in self.DEPRECATED

        formula = composition.reduced_formula

        entry_id = f"{self.__class__.__name__}-{formula}_{temperature}"

        self._validate_temperature(formula, temperature)
        self._temperature = temperature

        energy = self._get_energy(formula, temperature)

        super().__init__(
            composition,
            energy,
            energy_adjustments=energy_adjustments,
            data=data,
            entry_id=entry_id,
        )
        self._composition = composition
        self.name = formula
        if self.is_deprecated:
            self.name += " (deprecated)"

    def get_new_temperature(self, new_temperature: float) -> ExperimentalReferenceEntry:
        """Return a copy of the NISTReferenceEntry at the new specified temperature.

        Args:
            new_temperature: The new temperature to use [K]

        Returns:
            A copy of the NISTReferenceEntry at the new specified temperature.
        """
        new_entry_dict = self.as_dict()
        new_entry_dict["temperature"] = new_temperature

        return self.from_dict(new_entry_dict)

    def to_grand_entry(self, chempots: dict[Element, float]):
        """Convert an ExperimentalReferenceEntry to a GrandComputedEntry.

        Args:
            chempots: A dictionary of {element: chempot} pairs.

        Returns:
            A GrandComputedEntry.
        """
        return GrandPotPDEntry(self, chempots)

    def _validate_temperature(self, formula: str, temperature: float) -> None:
        """Ensure that the temperature is from a valid range."""
        if self.is_deprecated:
            logger.warning(f"{formula} is deprecated! Using a formation energy of 0.0 eV.")
            return  # skip validation

        if formula not in self.REFERENCES:
            raise ValueError(f"{formula} not in reference data!")

        g = self.REFERENCES[formula]
        if temperature < min(g) or temperature > max(g):
            raise ValueError(f"Temperature must be selected from range: {min(g)} K to {max(g)} K")

    def _get_energy(self, formula: str, temperature: float) -> float:
        """Convenience method for accessing and interpolating experimental data.

        Args:
            formula: Chemical formula by which to search experimental data.
            temperature: Absolute temperature [K].

        Returns:
            Gibbs free energy of formation of formula at specified temperature [eV]
        """
        if self.is_deprecated:
            return 0.0

        data = self.REFERENCES[formula]

        if temperature % 100 > 0:
            g_interp = interp1d(list(data.keys()), list(data.values()))
            return g_interp(temperature)[()]

        return data[temperature]

    @property
    def temperature(self) -> float:
        """Returns temperature used to calculate entry's energy."""
        return self._temperature

    @property
    def is_experimental(self) -> bool:
        """Returns True by default."""
        return True

    @property
    def is_element(self) -> bool:
        """Returns True if the entry is an element."""
        return self.composition.is_element

    @property
    def unique_id(self) -> str:
        """Returns a unique ID for the entry based on its entry-id and temperature. This is
        useful because the same entry-id can be used for multiple entries at different
        temperatures.
        """
        return self.entry_id

    def as_dict(self) -> dict:
        """Returns:
        A dict representation of the Entry.
        """
        d = super().as_dict()

        d["temperature"] = self.temperature

        del d["energy"]
        del d["entry_id"]
        del d["parameters"]
        del d["correction"]

        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentalReferenceEntry:
        """Generate an ExperimentalReferenceEntry object from a dictionary.

        Args:
            d: dictionary representation of the entry

        Returns:
            ExperimentalReferenceEntry object initialized with data from the dictionary.
        """
        dec = MontyDecoder()
        return cls(
            composition=Composition(d["composition"]),
            temperature=d["temperature"],
            energy_adjustments=dec.process_decoded(d["energy_adjustments"]),
            data=d["data"],
        )

    def __repr__(self) -> str:
        output = []
        if self.is_deprecated:
            output.append("DEPRECATED")

        output.extend(
            [
                f"{self.__class__.__name__} | {self.composition.reduced_formula}",
                f"Gibbs Energy ({self.temperature} K) = {self.energy:.4f}",
            ]
        )
        return "\n".join(output)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return (
                (self.composition.reduced_formula == other.composition.reduced_formula)
                and (self.temperature == other.temperature)
                and (set(self.energy_adjustments) == set(other.energy_adjustments))
            )
        return False

    def __hash__(self) -> int:
        data_md5 = hashlib.md5(  # nosec
            f"{self.__class__.__name__}{self.composition}_{self.temperature}".encode()
        ).hexdigest()
        return int(data_md5, 16)
