"""
Energy correction classes for entry objects.
"""

from pymatgen.entries.computed_entries import CompositionEnergyAdjustment

CARBONATE_CORRECTION = (
    0.82820  # eV per (CO3)2- anion in composition; see Jupyter NB for fitting
)


class CarbonateCorrection(CompositionEnergyAdjustment):
    """
    Correct carbonate energies to obtain the right formation energies.
    """

    def __init__(self, num_ions, carbonate_correction=CARBONATE_CORRECTION):
        """
        Initalizes a carbonate correction object

        Args:
            num_ions (int): Number of carbonate ions in the composition object
            carbonate_correction (float): Energy correction per atom per (CO3)2- anion
        """
        self._num_ions = num_ions
        self._carbonate_correction = carbonate_correction

        super().__init__(
            adj_per_atom=carbonate_correction,
            n_atoms=num_ions,
            name="Carbonate Correction",
            description=(
                "Correction for dGf with (CO3)2- anion, as fit to MP data (300 K)."
            ),
        )

    @property
    def num_ions(self):
        """
        Number of carbonate ions ion the composition object
        """
        return self._num_ions

    @property
    def carbonate_correction(self):
        """
        Energy correction for carbonate ion, eV per (CO3)2- anion
        """
        return self._carbonate_correction
