"""A computed entry object for estimating the Gibbs free energy of formation. Note that
this is similar to the implementation within pymatgen, but has been refactored here to
add extra functionality.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import GrandPotPDEntry
from pymatgen.entries.computed_entries import ComputedEntry, ConstantEnergyAdjustment
from scipy.interpolate import interp1d

from rxn_network.core import Composition
from rxn_network.data import G_ELEMS

if TYPE_CHECKING:
    from pymatgen.core.periodic_table import Element
    from pymatgen.core.structure import Structure
    from pymatgen.entries.computed_entries import EnergyAdjustment


class GibbsComputedEntry(ComputedEntry):
    """An extension to ComputedEntry which estimates the Gibbs free energy of formation
    of solids using energy adjustments from the machine-learned SISSO descriptor from
    Bartel et al. (2018).  Note that this is similar to the implementation within
    pymatgen, but has been refactored here to add extra functionality.

    WARNING: This descriptor only applies to solids. See entries.nist.NISTReferenceEntry
    for common gases (e.g. CO2).

    If you use this entry class in your work, please consider citing the following
    paper:

        Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R., Tumas, W., Weimer,
        A. W., … Holder, A. M. (2018). Physical descriptor for the Gibbs energy of
        inorganic crystalline solids and temperature-dependent materials chemistry.
        Nature Communications, 9(1), 4168. https://doi.org/10.1038/s41467-018-06682-4.
    """

    def __init__(
        self,
        composition: Composition,
        formation_energy_per_atom: float,
        volume_per_atom: float,
        temperature: float,
        energy_adjustments: list[EnergyAdjustment] | None = None,
        parameters: dict | None = None,
        data: dict | None = None,
        entry_id: object | None = None,
    ):
        """A new computed entry object is returned with a supplied energy correction
        representing the difference between the formation enthalpy at T=0K and the
        Gibbs formation energy at the specified temperature.

        Args:
            composition: The composition object (pymatgen)
            formation_energy_per_atom: Calculated formation enthalpy, dH, at T = 298 K,
                normalized to the total number of atoms in the composition. NOTE: since
                this is a _formation_ energy, it must be calculated using a phase
                diagram construction.
            volume_per_atom: The total volume of the associated structure divided by its
                number of atoms.
            temperature: Temperature [K] by which to acquire dGf(T); must be selected
                from a range of [300, 2000] K. If temperature is not selected from one
                of [300, 400, 500, ... 2000 K], then free energies will be interpolated.
            energy_adjustments: Optional list of energy adjustments.
            parameters: Optional list of calculation parameters.
            data: Optional dictionary containing entry data.
            entry_id: An identifying string for the entry, such as the entry's mpid
                (e.g., "mp-25025"). While optional, this is recommended to set as
                several downstream classes depend on its use. If an entry_id is not
                provided, a combination of the composition and volume will be used
                (e.g., "Li2O_8.4266").
        """
        composition = Composition(composition)
        self._composition = composition
        self.volume_per_atom = volume_per_atom
        num_atoms = composition.num_atoms
        if temperature < 300 or temperature > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        if energy_adjustments is not None:
            energy_adjustments = [
                adjustment for adjustment in energy_adjustments if adjustment.name != "Gibbs SISSO Correction"
            ]
        else:
            energy_adjustments = []

        energy_adjustments.append(
            ConstantEnergyAdjustment(
                self.gibbs_adjustment(temperature),
                uncertainty=0.05 * num_atoms,  # descriptor has ~50 meV/atom MAD
                name="Gibbs SISSO Correction",
                description=f"Gibbs correction: dGf({temperature} K) - dHf (298 K)",
            )
        )

        formation_energy = num_atoms * formation_energy_per_atom

        if entry_id is None:  # should set an entry_id for downstream processing
            entry_id = f"{composition.reduced_formula}_{volume_per_atom:.4f}"

        super().__init__(
            composition=composition,
            energy=formation_energy,
            energy_adjustments=energy_adjustments,
            parameters=parameters,
            data=data,
            entry_id=entry_id,
        )
        self._composition = composition
        self.formation_energy_per_atom = formation_energy_per_atom
        self.temperature = temperature

    def get_new_temperature(self, new_temperature: float) -> GibbsComputedEntry:
        """Return a copy of the GibbsComputedEntry at the new specified temperature.

        Args:
            new_temperature: The new temperature to use [K]

        Returns:
            A copy of the GibbsComputedEntry, initialized at the new specified
            temperature.
        """
        new_entry_dict = self.as_dict()
        new_entry_dict["temperature"] = new_temperature

        return self.from_dict(new_entry_dict)

    def gibbs_adjustment(self, temperature: float) -> float:
        """Returns the difference between the predicted Gibbs formation energy and the
        formation enthalpy at 298 K, i.e., dGf(T) - dHf(298 K). Calculated using
        SISSO descriptor from Bartel et al. (2018) and elemental chemical potentials
        (FactSage).

        Units: eV (not normalized)

        Args:
            temperature: The absolute temperature [K].

        Returns:
            The correction to Gibbs free energy of formation (eV) from DFT energy.
        """
        if self._composition.is_element:
            return 0

        num_atoms = self._composition.num_atoms
        reduced_mass = self._reduced_mass(self._composition)

        return num_atoms * self._g_delta_sisso(self.volume_per_atom, reduced_mass, temperature) - self._sum_g_i(
            self._composition, temperature
        )

    def to_grand_entry(self, chempots: dict[Element, float]) -> GrandPotPDEntry:
        """Convert a GibbsComputedEntry to a GrandComputedEntry.

        Args:
            chempots: A dictionary of {element: chempot} pairs.

        Returns:
            A GrandComputedEntry.
        """
        return GrandPotPDEntry(self, chempots)

    def copy(self) -> GibbsComputedEntry:
        """Returns a deepcopy of the GibbsComputedEntry object."""
        return deepcopy(self)

    @staticmethod
    def _g_delta_sisso(volume_per_atom: float, reduced_mass: float, temp: float) -> float:
        """G^delta as predicted by SISSO-learned descriptor from Eq. (4) in Bartel et al.
        (2018).

        Args:
            volume_per_atom: volume per atom [Å^3/atom]
            reduced_mass: reduced mass as calculated with pair-wise sum formula [amu]
            temp: Temperature [K]

        Returns:
            float: G^delta
        """
        return (
            (-2.48e-4 * np.log(volume_per_atom) - 8.94e-5 * reduced_mass / volume_per_atom) * temp
            + 0.181 * np.log(temp)
            - 0.882
        )

    @staticmethod
    def _sum_g_i(composition: Composition, temperature: float) -> float:
        """Sum of the stoichiometrically weighted chemical potentials [eV] of the elements
        at specified temperature, as acquired from data/elements.json.
        """
        elems = composition.get_el_amt_dict()

        if temperature % 100 > 0:
            sum_g_i = 0
            for elem, amt in elems.items():
                g_interp = interp1d(
                    [float(t) for t in G_ELEMS],
                    [g_dict[elem] for g_dict in G_ELEMS.values()],
                )
                sum_g_i += amt * g_interp(temperature)
        else:
            sum_g_i = sum(amt * G_ELEMS[str(temperature)][elem] for elem, amt in elems.items())

        return sum_g_i

    @staticmethod
    def _reduced_mass(composition: Composition) -> float:
        """Reduced mass [amu] as calculated via Eq. 6 in Bartel et al. (2018),
        to be used in SISSO descriptor equation.
        """
        reduced_comp = composition.reduced_composition
        num_elems = len(reduced_comp.elements)
        elem_dict = reduced_comp.get_el_amt_dict()

        denominator = (num_elems - 1) * reduced_comp.num_atoms

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0.0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j)

        return (1 / denominator) * mass_sum

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        formation_energy_per_atom: float,
        temperature: float,
        **kwargs,
    ) -> GibbsComputedEntry:
        """Constructor method for building a GibbsComputedEntry from a structure,
        formation enthalpy, and temperature.

        Args:
            structure: Structure object (pymatgen)
            formation_energy_per_atom: Formation enthalpy at T = 298 K associated
                with structure
            temperature: Desired temperature [K] for acquiring dGf(T)
            **kwargs: Optional kwargs to be passed to GibbsComputedEntry.__init__, such
                as entry_id, energy_adjustments, parameters, and data.

        Returns:
            A new GibbsComputedEntry object
        """
        composition = Composition(structure.composition).element_composition
        volume_per_atom = structure.volume / structure.num_sites
        return cls(
            composition=composition,
            formation_energy_per_atom=formation_energy_per_atom,
            volume_per_atom=volume_per_atom,
            temperature=temperature,
            **kwargs,
        )

    @property
    def is_experimental(self) -> bool:
        """Returns True if self.data contains {"theoretical": False}. If
        theoretical is not specified but there is greater than 1 icsd_id provided,
        assumes that the presence of an icsd_id means the entry is experimental.
        """
        if "theoretical" in self.data:
            return not self.data["theoretical"]
        if "icsd_ids" in self.data:
            return len(self.data["icsd_ids"]) >= 1

        return False

    @property
    def unique_id(self) -> str:
        """Returns a unique ID for the entry based on its entry-id and temperature. This is
        useful because the same entry-id can be used for multiple entries at different
        temperatures.
        """
        return f"{self.entry_id}_{self.temperature}"

    def as_dict(self) -> dict:
        """Returns an MSONable dict."""
        data = super().as_dict()
        data["volume_per_atom"] = self.volume_per_atom
        data["formation_energy_per_atom"] = self.formation_energy_per_atom
        data["temperature"] = self.temperature
        return data

    @classmethod
    def from_dict(cls, d: dict) -> GibbsComputedEntry:
        """Returns a GibbsComputedEntry object from MSONable dictionary."""
        dec = MontyDecoder()
        return cls(
            composition=d["composition"],
            formation_energy_per_atom=d["formation_energy_per_atom"],
            volume_per_atom=d["volume_per_atom"],
            temperature=d["temperature"],
            energy_adjustments=dec.process_decoded(d["energy_adjustments"]),
            parameters=d["parameters"],
            data=d["data"],
            entry_id=d["entry_id"],
        )

    def __repr__(self):
        output = [
            (
                f"GibbsComputedEntry | {self.entry_id} | {self.composition.formula} "
                f"({self.composition.reduced_formula})"
            ),
            f"Gibbs Energy ({self.temperature} K) = {self.energy:.4f}",
        ]
        return "\n".join(output)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False

        if not np.isclose(self.temperature, other.temperature):
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
                self.temperature,
            )
        )
