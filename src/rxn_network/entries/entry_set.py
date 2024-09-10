"""An entry set class for automatically building GibbsComputedEntry objects. Some of this
code has been adapted from the EntrySet class in pymatgen.
"""

from __future__ import annotations

import collections
import inspect
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from monty.json import MontyDecoder, MSONable
from monty.serialization import loadfn
from numpy.random import normal
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Element
from pymatgen.entries.computed_entries import ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
from tqdm import tqdm

from rxn_network.core import Composition
from rxn_network.data import PATH_TO_NIST
from rxn_network.entries.corrections import (
    CarbonateCorrection,
    CarbonDioxideAtmosphericCorrection,
)
from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.entries.freed import FREEDReferenceEntry
from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.entries.interpolated import InterpolatedEntry
from rxn_network.entries.nist import NISTReferenceEntry
from rxn_network.thermo.utils import expand_pd
from rxn_network.utils.funcs import get_logger

logger = get_logger(__name__)

#  prefer computed data for solids with high melting points (T >= 1500 ºC)
IGNORE_NIST_SOLIDS = loadfn(PATH_TO_NIST / "ignore_solids.json")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymatgen.entries.computed_entries import (
        ComputedEntry,
        ComputedStructureEntry,
        EnergyAdjustment,
    )


class GibbsEntrySet(collections.abc.MutableSet, MSONable):
    """This object is based on pymatgen's EntrySet class and includes factory methods for
    constructing GibbsComputedEntry objects from "zero-temperature"
    ComputedStructureEntry objects. It also offers convenient methods for acquiring
    entries from the entry set, whether that be using composition, stability, chemical
    system, etc.
    """

    def __init__(
        self,
        entries: Iterable[GibbsComputedEntry | ExperimentalReferenceEntry | InterpolatedEntry],
        calculate_e_above_hulls: bool = False,
        minimize_obj_size: bool = False,
    ):
        """The supplied collection of entries will automatically be converted to a set of
        unique entries.

        Args:
            entries: A collection of entry objects that will make up the entry set.
            calculate_e_above_hulls: Whether to pre-calculate the energy above hull for
                each entry and store that in that entry's data. Defaults to False.
            minimize_obj_size: Whether to reduce the size of the entry set by
                removing metadata from each entry. This may be useful when working with
                extremely large entry sets (or ComputedReaction sets). Defaults to
                False.
        """
        self.entries = set(entries)
        self.calculate_e_above_hulls = calculate_e_above_hulls
        self.minimize_obj_size = minimize_obj_size

        if minimize_obj_size:
            for e in self.entries:
                e.parameters = {}
                e.data = {}
        if calculate_e_above_hulls:
            for e in self.entries:
                e.data["e_above_hull"] = self.get_e_above_hull(e)

    def __contains__(self, item) -> bool:
        return item in self.entries

    def __iter__(self):
        return self.entries.__iter__()

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: GibbsComputedEntry | ExperimentalReferenceEntry | InterpolatedEntry) -> None:
        """Add an entry to the set. This is an IN-PLACE method.

        Args:
            entry: An entry object.
        """
        self.entries.add(entry)
        self._clear_cache()

    def update(
        self,
        entries: Iterable[GibbsComputedEntry | ExperimentalReferenceEntry | InterpolatedEntry],
    ) -> None:
        """Add an iterable of entries to the set. This is an IN-PLACE method.

        Args:
            entries: Iterable of entry objects to add to the set.
        """
        self.entries.update(entries)
        self._clear_cache()

    def discard(self, entry: GibbsComputedEntry | ExperimentalReferenceEntry) -> None:
        """Discard an entry. This is an IN-PLACE method.

        Args:
            entry: An entry object.
        """
        self.entries.discard(entry)
        self._clear_cache()

    @cached_property
    def pd_dict(self) -> dict:
        """Returns a dictionary of phase diagrams, keyed by the chemical system. This is
        acquired using the helper method expand_pd() and represents one of the simplest
        divisions of sub-PDs for large chemical systems. Cached for speed.
        """
        return expand_pd(self.entries)

    def get_subset_in_chemsys(self, chemsys: list[str] | str) -> GibbsEntrySet:
        """Returns a GibbsEntrySet containing only the set of entries belonging to
        a particular chemical system (including subsystems). For example, if the entries
        are from the Li-Fe-P-O system, and chemsys=["Li", "O"], only the Li, O, and Li-O
        entries are returned.

        Args:
            chemsys: Chemical system specified as list of elements. E.g., ["Li", "O"]

        Returns: GibbsEntrySet
        """
        if isinstance(chemsys, str):
            chemsys = chemsys.split("-")
        chem_sys = set(chemsys)
        if not chem_sys.issubset(self.chemsys):
            raise ValueError(f"{chem_sys} is not a subset of {self.chemsys}")

        subset = set()
        for e in self.entries:
            elements = [sp.symbol for sp in e.composition]
            if chem_sys.issuperset(elements):
                subset.add(e)

        return GibbsEntrySet(subset, calculate_e_above_hulls=False)

    def filter_by_stability(self, e_above_hull: float, include_polymorphs: bool | None = False) -> GibbsEntrySet:
        """Filter the entry set by a metastability (energy above hull) cutoff.

        Args:
            e_above_hull: Energy above hull, the cutoff describing the allowed
                metastability of the entries as determined via phase diagram
                construction.
            include_polymorphs: optional specification of whether to include
                metastable polymorphs. Defaults to False.

        Returns:
            A new GibbsEntrySet where the entries have been filtered by an energy
            cutoff (e_above_hull) via phase diagram construction.
        """
        pd_dict = self.pd_dict

        filtered_entries: set[GibbsComputedEntry | NISTReferenceEntry] = set()
        all_comps: dict[str, GibbsComputedEntry | NISTReferenceEntry] = {}

        for pd in pd_dict.values():
            for entry in pd.all_entries:
                if entry in filtered_entries or pd.get_e_above_hull(entry) > e_above_hull:
                    continue

                formula = entry.composition.reduced_formula
                if not include_polymorphs and (formula in all_comps):
                    if all_comps[formula].energy_per_atom < entry.energy_per_atom:
                        continue
                    filtered_entries.remove(all_comps[formula])

                all_comps[formula] = entry
                filtered_entries.add(entry)

        return self.__class__(list(filtered_entries))

    def build_indices(self) -> None:
        """Builds the indices for the entry set in place. This method is called whenever an
        entry is added/removed the entry set. The entry indices are useful for querying
        the entry set for specific entries.

        Warning: this internally modifies the entries in the entry set by updating data
        for each entry to include the index.

        Returns:
            None
        """
        for idx, e in enumerate(self.entries_list):
            e.data.update({"idx": idx})

    def get_min_entry_by_formula(self, formula: str) -> ComputedEntry:
        """Helper method for acquiring the ground state entry with the specified formula.

        Args:
            formula: The chemical formula of the desired entry.

        Returns:
            Ground state computed entry object.
        """
        return self.min_entries_by_formula[Composition(formula).reduced_formula]

    def get_stabilized_entry(self, entry: ComputedEntry, tol: float = 1e-3, force=False) -> ComputedEntry:
        """Helper method for lowering the energy of a single entry such that it is just
        stable on the phase diagram. If the entry is already stable, it will be
        returned unchanged.

        Note: if the entry includes the "e_above_hull" data, this value will be used to
        stabilize the entry. Otherwise, the energy above hull will be calculated via
        creation of a phase diagram; this can take a long time for repeated calls.

        Args:
            entry: A computed entry object.
            tol: The numerical padding added to the energy correction to guarantee
                that it is determined to be stable during phase diagram construction.
            force: due to numerical stability issues, if the entry is very close to the
                hull (i.e., e_above_hull > 0 but very small), it may not be stabilized.
                This option forces stabilization even if e_above_hull evalutes to
                "zero". This can be crucial for certain edge cases.

        Returns:
            A new ComputedEntry with energy adjustment making it appear to be stable
            with the current entry data (i.e., compositional phase diagram).
        """
        e_above_hull = None
        if hasattr(entry, "data"):
            e_above_hull = entry.data.get("e_above_hull")

        if e_above_hull is None:
            e_above_hull = self.get_e_above_hull(entry)

        if e_above_hull == 0.0 and not force:
            new_entry = entry
        else:
            e_adj = -1 * e_above_hull * entry.composition.num_atoms - tol
            adjustment = ConstantEnergyAdjustment(
                value=e_adj,
                name="Stabilization Adjustment",
                description="Shifts energy so that entry is on the convex hull",
            )
            new_entry = self.get_adjusted_entry(entry, adjustment)

        return new_entry

    def get_entries_with_new_temperature(self, new_temperature: float) -> GibbsEntrySet:
        """Returns a new GibbsEntrySet with entries that have had their energies
        modified by using a new temperature.

        Note: this will clear the "e_above_hull" data for each entry and re-calculate
        them only if the original entry set had them calculated.
        """
        new_entries = []

        for entry in self.entries_list:
            try:
                new_entry = entry.get_new_temperature(new_temperature)
            except ValueError as e:
                logger.warning(f"Could not get new temperature for entry: {entry}. {e}")
                continue

            new_entry.data["e_above_hull"] = None
            new_entries.append(new_entry)

        return self.__class__(new_entries, calculate_e_above_hulls=self.calculate_e_above_hulls)

    def get_entries_with_jitter(self) -> GibbsEntrySet:
        """Returns a new GibbsEntrySet with entries that have had their energies shifted by
        randomly sampled noise to account for uncertainty in data. This is done by
        sampling from a Gaussian distribution using the entry's "correction_uncertainty"
        attribute as the scale.

        Returns:
            A new GibbsEntrySet with entries that have had their energies shifted by
            random Gaussian noise based on their "correction_uncertainty" values.
        """
        entries = deepcopy(self.entries_list)
        new_entries = []
        jitter = normal(size=len(entries))

        for idx, entry in enumerate(entries):
            if entry.is_element:
                continue
            adj = ConstantEnergyAdjustment(
                value=jitter[idx] * entry.correction_uncertainty,
                name="Random jitter",
                description=("Randomly sampled (Gaussian) noise to account for uncertainty in data"),
            )
            new_entries.append(self.get_adjusted_entry(entry, adj))

        return GibbsEntrySet(new_entries)

    def get_interpolated_entry(self, formula: str, tol_per_atom: float = 1e-3) -> ComputedEntry:
        """Helper method for interpolating an entry from the entry set.

        Args:
            formula: The chemical formula of the desired entry.
            tol_per_atom: the energy shift (eV/atom) below the hull energy so that the
                interpolated entry is guaranteed to be stable. Defaults to 1 meV/atom.

        Returns:
            An interpolated GibbsComputedEntry object.
        """
        comp = Composition(formula).reduced_composition
        pd_entries = self.get_subset_in_chemsys([str(e) for e in comp.elements])

        energy = PhaseDiagram(pd_entries).get_hull_energy(comp) - tol_per_atom * comp.num_atoms

        adj = ConstantEnergyAdjustment(  # for keeping track of uncertainty
            value=0.0,
            uncertainty=0.05 * comp.num_atoms,  # conservative: 50 meV/atom uncertainty
            name="Interpolation adjustment (for uncertainty)",
            description="Maintains uncertainty due to interpolation",
        )

        return InterpolatedEntry(
            comp,
            energy,
            energy_adjustments=[adj],
            entry_id=f"InterpolatedEntry-{comp.formula}_{self.temperature}",
        )

    def get_e_above_hull(self, entry: ComputedEntry) -> float:
        """Helper method for calculating the energy above hull for a single entry.

        Args:
            entry: A ComputedEntry object.

        Returns:
            The energy above hull for the entry.
        """
        for chemsys, pd in self.pd_dict.items():
            elems_pd = set(chemsys.split("-"))
            elems_entry = set(entry.composition.chemical_system.split("-"))

            if elems_entry.issubset(elems_pd):
                return pd.get_e_above_hull(entry)

        raise ValueError("Entry not in any of the phase diagrams in pd_dict!")

    @classmethod
    def from_pd(
        cls,
        pd: PhaseDiagram,
        temperature: float,
        include_nist_data: bool = True,
        include_freed_data: bool = False,
        apply_carbonate_correction: bool = True,
        apply_atmospheric_co2_correction: bool = True,
        ignore_nist_solids: bool = True,
        calculate_e_above_hulls: bool = False,
        minimize_obj_size: bool = False,
    ) -> GibbsEntrySet:
        """Constructor method for building a GibbsEntrySet from an existing phase diagram.

        Args:
            pd: Phase Diagram object (pymatgen)
            temperature: Temperature [K] for determining Gibbs Free Energy of
                formation, dGf(T)
            include_nist_data: Whether to include NIST data in the entry set. Defaults
                to True.
            include_freed_data: Whether to include Freed data in the entry set. Defaults
                to False. Use at your own risk!
            apply_carbonate_correction: Whether to apply the fit energy
                correction for carbonates. Defaults to True.
            apply_atmospheric_co2_correction: Whether to modify the chemical potential
                of CO2 by its partial pressure in the atmosphere (0.04%). Defaults to True.
            ignore_nist_solids: Whether to ignore NIST data for the solids specified in
                the "data/nist/ignore_solids.json" file; these all have melting points
                Tm >= 1500 ºC. Defaults to Ture.
            calculate_e_above_hulls: Whether to calculate energy above hull for each
                entry and store in the entry's data. Defaults to False.
            minimize_obj_size: Whether to minimize the size of the object by removing
                unnecessary attributes from the entries. Defaults to False.

        Returns:
            A GibbsEntrySet containing a collection of GibbsComputedEntry and
            experimental reference entry objects at the specified temperature.

        """
        gibbs_entries = []
        experimental_formulas = []

        for entry in pd.all_entries:
            composition = entry.composition
            formula = composition.reduced_formula

            if composition.is_element and entry not in pd.el_refs.values() or formula in experimental_formulas:
                continue

            new_entries = []
            new_entry = None
            if include_nist_data:
                new_entry = cls._check_for_experimental(
                    formula, "nist", temperature, ignore_nist_solids, apply_atmospheric_co2_correction
                )
                if new_entry:
                    new_entries.append(new_entry)

            if include_freed_data:
                new_entry = cls._check_for_experimental(
                    formula, "freed", temperature, ignore_nist_solids, apply_atmospheric_co2_correction
                )
                if new_entry:
                    new_entries.append(new_entry)

            if new_entry:
                experimental_formulas.append(formula)
            else:
                energy_adjustments = []
                if apply_carbonate_correction:
                    corr = cls._get_carbonate_correction(entry)
                    if corr is not None:
                        energy_adjustments.append(corr)
                if apply_atmospheric_co2_correction and formula == "CO2":
                    energy_adjustments.append(
                        CarbonDioxideAtmosphericCorrection(entry.composition.num_atoms, temperature)
                    )

                structure = entry.structure
                formation_energy_per_atom = pd.get_form_energy_per_atom(entry)

                gibbs_entry = GibbsComputedEntry.from_structure(
                    structure=structure,
                    formation_energy_per_atom=formation_energy_per_atom,
                    temperature=temperature,
                    energy_adjustments=energy_adjustments,
                    parameters=entry.parameters,
                    data=entry.data,
                    entry_id=entry.entry_id,
                )

                new_entries.append(gibbs_entry)

            gibbs_entries.extend(new_entries)

        return cls(
            gibbs_entries,
            calculate_e_above_hulls=calculate_e_above_hulls,
            minimize_obj_size=minimize_obj_size,
        )

    def copy(self) -> GibbsEntrySet:
        """Returns a copy of the entry set."""
        return GibbsEntrySet(entries=self.entries)

    def as_dict(self) -> dict:
        """Returns a JSON serializable dict representation of the entry set."""
        d = super().as_dict()
        d["entries"] = [e.as_dict() for e in self.entries]
        d["calculate_e_above_hulls"] = self.calculate_e_above_hulls
        return d

    @classmethod
    def from_computed_entries(
        cls,
        entries: Iterable[ComputedStructureEntry],
        temperature: float,
        include_nist_data: bool = True,
        include_freed_data: bool = False,
        apply_carbonate_correction: bool = True,
        apply_atmospheric_co2_correction: bool = True,
        ignore_nist_solids: bool = True,
        calculate_e_above_hulls: bool = False,
        minimize_obj_size: bool = False,
    ) -> GibbsEntrySet:
        """Constructor method for initializing GibbsEntrySet from T = 0 K
        ComputedStructureEntry objects, as acquired from a thermochemical database
        (e.g., The Materials Project).

        Automatically expands the phase diagram for large chemical systems (10 or more
        elements) to avoid limitations of Qhull.

        Args:
            entries: Iterable of ComputedStructureEntry objects. These can be downloaded
                from The Materials Project API or created manually with pymatgen.
            temperature: Temperature [K] for determining Gibbs Free Energy of
                formation, dGf(T)
            include_nist_data: Whether to include NIST-JANAF data in the entry set.
                Defaults to True.
            include_freed_data: Whether to include FREED data in the entry set. Defaults
                to False. WARNING: This dataset has not been thoroughly tested. Use at
                your own risk!
            apply_carbonate_correction: Whether to apply the fit GGA energy correction
                for carbonates. Defaults to True.
            apply_atmospheric_co2_correction: Whether to modify the chemical potential
                of CO2 by its partial pressure in the atmosphere (). Defaults to True.
            ignore_nist_solids: Whether to ignore NIST data for the solids specified in
                the "data/nist/ignore_solids.json" file; these all have melting points
                Tm >= 1500 ºC. Defaults to True.
            calculate_e_above_hulls: Whether to calculate energy above hull for each
                entry and store in the entry's data. Defaults to False.
            minimize_obj_size: Whether to minimize the size of the object by removing
                unrequired attributes from the entries. Defaults to False.

        Returns:
            A GibbsEntrySet containing a collection of GibbsComputedEntry and
            experimental reference entry objects at the specified temperature.
        """
        e_set = EntrySet(entries)
        new_entries: set[GibbsComputedEntry] = set()

        if len(e_set.chemsys) <= 9:  # Qhull algorithm struggles beyond 9 dimensions
            pd = PhaseDiagram(e_set)
            return cls.from_pd(
                pd,
                temperature,
                include_nist_data=include_nist_data,
                include_freed_data=include_freed_data,
                apply_carbonate_correction=apply_carbonate_correction,
                apply_atmospheric_co2_correction=apply_atmospheric_co2_correction,
                ignore_nist_solids=ignore_nist_solids,
                calculate_e_above_hulls=calculate_e_above_hulls,
                minimize_obj_size=minimize_obj_size,
            )

        pd_dict = expand_pd(list(e_set))
        logger.info("Building entries from expanded phase diagrams...")
        for _, pd in tqdm(pd_dict.items(), desc="GibbsComputedEntry"):
            gibbs_set = cls.from_pd(
                pd,
                temperature,
                include_nist_data=include_nist_data,
                include_freed_data=include_freed_data,
                apply_carbonate_correction=apply_carbonate_correction,
                apply_atmospheric_co2_correction=apply_atmospheric_co2_correction,
                ignore_nist_solids=ignore_nist_solids,
                calculate_e_above_hulls=calculate_e_above_hulls,
                minimize_obj_size=minimize_obj_size,
            )
            new_entries.update(gibbs_set)

        return cls(list(new_entries))

    @cached_property
    def entries_list(self) -> list[ComputedEntry]:
        """Returns a list of all entries in the entry set."""
        return sorted(self.entries, key=lambda e: e.composition)

    @cached_property
    def min_entries_by_formula(self) -> dict[str, ComputedEntry]:
        """Returns a dict of minimum energy entries in the entry set, indexed by
        formula.
        """
        min_entries = {}
        for e in self.entries:
            formula = e.composition.reduced_formula
            if formula not in min_entries:
                entries = filter(lambda x: x.composition.reduced_formula == formula, self.entries)
                min_entries[formula] = sorted(entries, key=lambda x: x.energy_per_atom)[0]

        return min_entries

    @cached_property
    def temperature(self) -> float:
        """Returns the temperature of entries in the dataset. More precisely, this is the
        temperature encountered when iterating over the dataset.

        Use at your own risk if mixing GibbsComputedEntry objects calculated at
        different temperatures -- this poses even more theoretical problems!
        """
        temp = 0.0
        for e in self.entries:
            if isinstance(e, (ExperimentalReferenceEntry, GibbsComputedEntry)):
                temp = e.temperature  # get temperature from any entry
                break
        return temp

    @cached_property
    def chemsys(self) -> set[str]:
        """Returns:
        Set of symbols representing the chemical system, e.g., {"Li", "Fe", "P",
        "O"}.
        """
        chemsys = set()
        for e in self.entries:
            chemsys.update([el.symbol for el in e.composition])
        return chemsys

    @staticmethod
    def get_adjusted_entry(entry: GibbsComputedEntry, adjustment: EnergyAdjustment) -> GibbsComputedEntry:
        """Gets an entry with an energy adjustment applied.

        Args:
            entry: A GibbsComputedEntry object.
            adjustment: An EnergyAdjustment object to apply to the entry.

        Returns:
            A new GibbsComputedEntry object with the energy adjustment applied.
        """
        entry_dict = entry.as_dict()
        original_entry = entry_dict.get("entry", None)

        if original_entry:
            energy_adjustments = original_entry["energy_adjustments"]
        else:
            energy_adjustments = entry_dict["energy_adjustments"]

        energy_adjustments.append(adjustment.as_dict())
        return MontyDecoder().process_decoded(entry_dict)

    @staticmethod
    def _check_for_experimental(
        formula: str,
        cls_name: str,
        temperature: float,
        ignore_nist_solids: bool,
        apply_atmospheric_co2_correction: bool,
    ):
        cls_name = cls_name.lower()
        if cls_name in ("nist", "nistreferenceentry"):
            cl = NISTReferenceEntry
        elif cls_name in ("freed", "freedreferenceentry"):
            cl = FREEDReferenceEntry
        else:
            raise ValueError("Invalid class name for experimental reference entry.")

        entry = None
        if formula in cl.REFERENCES:
            if cl == NISTReferenceEntry and ignore_nist_solids and formula in IGNORE_NIST_SOLIDS:
                return None

            energy_adjustments = None
            if apply_atmospheric_co2_correction and formula == "CO2":
                energy_adjustments = [CarbonDioxideAtmosphericCorrection(3, temperature)]

            try:
                entry = cl(
                    composition=Composition(formula), temperature=temperature, energy_adjustments=energy_adjustments
                )
            except ValueError as error:
                logger.debug(f"Compound {formula} is in {cl} tables but at different temperatures!: {error}")

        return entry

    @staticmethod
    def _get_carbonate_correction(entry):
        """Helper method for determining the carbonate correction for an entry.

        WARNING: The standard correction value provided in this module has been fit only
        to MP-derived entries (i.e., entries calculated with MPRelaxSet and MPStaticSet
        in VASP). Please check that the correction is valid for your entry.
        """
        comp = entry.composition

        if not {Element("C"), Element("O")}.issubset(comp.elements):
            return None

        if entry.parameters.get("run_type", None) not in ["GGA", "GGA+U"]:
            return None

        el_amts = comp.get_el_amt_dict()  # now get number of carbonate ions

        num_c = el_amts.get("C", 0)
        num_o = el_amts.get("O", 0)

        if num_c == 0 or num_o == 0:
            return None

        if not np.isclose(num_o / num_c, 3):
            return None

        return CarbonateCorrection(num_c)

    def _clear_cache(self) -> None:
        """Clears all cached properties. This method is called whenever the entry set is
        modified in place (as is done with the add method, etc.).
        """
        for name, value in inspect.getmembers(GibbsEntrySet):
            if isinstance(value, cached_property):
                try:
                    delattr(self, name)
                except AttributeError:
                    continue
