"""Energy correction classes for entry objects."""

from __future__ import annotations

import math

from pymatgen.entries.computed_entries import CompositionEnergyAdjustment

CARBONATE_CORRECTION = 0.830  # eV per (CO3)2- anion in composition; see Jupyter NB for fitting
PCO2 = 0.0004  # partial pressure of CO2 in the atmosphere (it's around 0.04%)


class CarbonateCorrection(CompositionEnergyAdjustment):
    """Supplies a carbonate correction due to systematic GGA errors in carbonate formation
    energies.

    See provided jupyter NB for fitting of the correction:
    data/fit_carbonate_correction.ipynb
    """

    def __init__(self, num_ions: int, carbonate_correction: float = CARBONATE_CORRECTION):
        """Initalizes a carbonate correction object.

        Args:
            num_ions: Number of carbonate ions in the composition object
            carbonate_correction: Energy correction, eV per (CO3)2- anion
        """
        self._num_ions = num_ions
        self._carbonate_correction = carbonate_correction

        super().__init__(
            adj_per_atom=carbonate_correction,
            n_atoms=num_ions,
            name="Carbonate Correction",
            description=("Correction for dGf with (CO3)2- anion, as fit to MP data (300 K)."),
        )

    @property
    def num_ions(self) -> int:
        """Number of carbonate ions ion the composition object."""
        return self._num_ions

    @property
    def carbonate_correction(self) -> float:
        """Energy correction for carbonate ion, eV per (CO3)2- anion."""
        return self._carbonate_correction


class CarbonDioxideAtmosphericCorrection(CompositionEnergyAdjustment):
    """Supplies a correction to the energy of CO2 due to its low partial pressure in the
    standard atmosphere (0.04%).
    """

    def __init__(self, n_atoms: int, temp: float, pco2: float = PCO2):
        """Initalizes a carbon dioxide correction object.

        Args:
            n_atoms: Number of atoms in the composition object
            temp: Temperature at which the correction is applied, in K
            pco2: Partial pressure of CO2 in the atmosphere, in atm
        """
        self._pco2 = pco2
        self._temp = temp

        super().__init__(
            adj_per_atom=self.get_dmu(),
            n_atoms=n_atoms,
            name="Atmospheric CO2 Correction",
            description=("Correction for CO2 energy based on partial pressure in the atmosphere"),
        )

    def get_dmu(self) -> float:
        """Returns the change in chemical potential of CO2 due to the atmospheric pCO2 at a given
        temperature.

        This is calculated by the formula: dmu = (1/3)*kTlnP(CO2).

        The factor of 1/3 accounts for the number of atoms in one formula unit of CO2,
        since the correction must be in eV/atom.
        """
        return (1 / 3) * 8.617e-5 * self.temp * math.log(self.pco2)

    @property
    def temp(self) -> float:
        """Temperature at which the correction is applied, in K."""
        return self._temp

    @property
    def pco2(self) -> float:
        """Partial pressure of CO2 in the atmosphere, in atm."""
        return self._pco2
