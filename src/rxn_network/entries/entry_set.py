"""
An entry set class for automatically building GibbsComputedEntry objects. Some of this
code has been adapted from the EntrySet class in pymatgen.
"""
import collections
import logging
import warnings
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Union

from monty.dev import deprecated
from monty.json import MontyDecoder, MSONable
from numpy.random import normal
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,
    ConstantEnergyAdjustment,
)
from pymatgen.entries.entry_tools import EntrySet
from tqdm.auto import tqdm

from rxn_network.entries.barin import BarinReferenceEntry
from rxn_network.entries.experimental import ExperimentalReferenceEntry
from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.entries.nist import NISTReferenceEntry
from rxn_network.thermo.utils import expand_pd


class GibbsEntrySet(collections.abc.MutableSet, MSONable):
    """
    This object is based on pymatgen's EntrySet class and includes factory methods for constructing
    GibbsComputedEntry objects from zero-temperature ComputedStructureEntry objects. It
    also offers convenient methods for acquiring entries from the entry set, whether
    that be using composition, stability, chemical system, etc.
    """

    def __init__(
        self, entries: Iterable[Union[GibbsComputedEntry, ExperimentalReferenceEntry]]
    ):
        """
        The supplied collection of entries will automatically be converted to a set of
        unique entries.

        Args:
            entries: A collection of entry objects that will make up the entry set.
        """
        self.entries = set(entries)

    def __contains__(self, item):
        return item in self.entries

    def __iter__(self):
        return self.entries.__iter__()

    def __len__(self):
        return len(self.entries)

    def add(self, entry: Union[GibbsComputedEntry, ExperimentalReferenceEntry]):
        """
        Add an entry to the set.

        :param element: Entry
        """
        self.entries.add(entry)

    def discard(self, entry: Union[GibbsComputedEntry, ExperimentalReferenceEntry]):
        """
        Discard an entry.

        :param element: Entry
        """
        self.entries.discard(entry)

    def get_subset_in_chemsys(self, chemsys: List[str]) -> "GibbsEntrySet":
        """
        Returns a GibbsEntrySet containing only the set of entries belonging to
        a particular chemical system (including subsystems). For example, if the entries
        are from the Li-Fe-P-O system, and chemsys=["Li", "O"], only the Li, O, and Li-O
        entries are returned.

        Args:
            chemsys: Chemical system specified as list of elements. E.g., ["Li", "O"]

        Returns: GibbsEntrySet
        """
        chem_sys = set(chemsys)
        if not chem_sys.issubset(self.chemsys):
            raise ValueError(f"{chem_sys} is not a subset of {self.chemsys}")
        subset = set()
        for e in self.entries:
            elements = [sp.symbol for sp in e.composition.keys()]
            if chem_sys.issuperset(elements):
                subset.add(e)

        return GibbsEntrySet(subset)

    def filter_by_stability(
        self, e_above_hull: float, include_polymorphs: Optional[bool] = False
    ) -> "GibbsEntrySet":
        """
        Filter the entry set by a metastability (energy above hull) cutoff.

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
        pd_dict = expand_pd(self.entries)

        filtered_entries: Set[Union[GibbsComputedEntry, NISTReferenceEntry]] = set()
        all_comps: Dict[str, Union[GibbsComputedEntry, NISTReferenceEntry]] = {}

        for _, pd in pd_dict.items():
            for entry in pd.all_entries:
                if (
                    entry in filtered_entries
                    or pd.get_e_above_hull(entry) > e_above_hull
                ):
                    continue

                formula = entry.composition.reduced_formula
                if not include_polymorphs and (formula in all_comps):
                    if all_comps[formula].energy_per_atom < entry.energy_per_atom:
                        continue
                    filtered_entries.remove(all_comps[formula])

                all_comps[formula] = entry
                filtered_entries.add(entry)

        return self.__class__(list(filtered_entries))

    def build_indices(self):
        """
        Builds the indices for the entry set. This method is called whenever an entry is
        added/removed the entry set. The entry indices are useful for querying the entry
        set for specific entries.

        Warning: this internally modifies the entries in the entry set by updating data
        for each entry to include the index.

        Returns:
            None
        """
        for idx, e in enumerate(self.entries_list):
            e.data.update({"idx": idx})

    def get_min_entry_by_formula(self, formula: str) -> ComputedEntry:
        """
        Helper method for acquiring the ground state entry with the specified formula.

        Args:
            formula: The chemical formula of the desired entry.

        Returns:
            Ground state computed entry object.
        """
        comp = Composition(formula).reduced_composition
        possible_entries = filter(
            lambda x: x.composition.reduced_composition == comp, self.entries
        )
        return sorted(possible_entries, key=lambda x: x.energy_per_atom)[0]

    def get_stabilized_entry(
        self, entry: ComputedEntry, tol: float = 1e-6
    ) -> ComputedEntry:
        """
        Helper method for lowering the energy of a single entry such that it is just
        barely stable on the phase diagram.

        Args:
            entry: A computed entry object.
            tol: The numerical padding added to the energy correction to guarantee
                that it is determined to be stable during phase diagram construction.

        Returns:
            A new ComputedEntry with energy adjustment making it appear to be stable.
        """
        chemsys = [str(e) for e in entry.composition.elements]
        entries = self.get_subset_in_chemsys(chemsys)
        pd = PhaseDiagram(entries)
        e_above_hull = pd.get_e_above_hull(entry)

        if e_above_hull == 0.0:
            new_entry = entry
        else:
            e_adj = -1 * pd.get_e_above_hull(entry) * entry.composition.num_atoms - tol
            adjustment = ConstantEnergyAdjustment(
                value=e_adj,
                name="Stabilization Adjustment",
                description="Shifts energy so that " "entry is on the convex hull",
            )

            entry_dict = entry.as_dict()
            entry_dict["energy_adjustments"].append(adjustment)
            new_entry = MontyDecoder().process_decoded(entry_dict)

        return new_entry

    @deprecated(
        get_stabilized_entry,
        "This method has been renamed. Use get_stabilized_entry instead.",
    )
    def stabilize_entry(self, entry: ComputedEntry, tol: float = 1e-6) -> ComputedEntry:
        """
        This method is deprecated. Use get_stabilized_entry instead.
        """
        return self.get_stabilized_entry(entry, tol)

    def get_entries_with_jitter(self) -> "GibbsEntrySet":
        """
        Returns a new GibbsEntrySet with entries that have had their energies shifted by
        randomly sampled noise to account for uncertainty in data. This is done by
        sampling from a Gaussian distribution using the entry's "correction_uncertainty"
        attribute as the scale.

        Args:
            None
        Returns:
            A new GibbsEntrySet with entries that have had their energies shifted by
            random noise.
        """
        entries = deepcopy(self.entries_list)
        jitter = normal(size=len(entries))

        for idx, entry in enumerate(entries):
            if entry.is_element:
                continue
            adj = ConstantEnergyAdjustment(
                value=jitter[idx] * entry.correction_uncertainty,
                name="Random jitter",
                description="Randomly sampled noise to account for uncertainty in data",
            )
            entry.energy_adjustments.append(adj)

        return GibbsEntrySet(entries)

    def get_interpolated_entry(self, formula: str, tol=1e-6) -> ComputedEntry:
        """
        Helper method for interpolating an entry from the entry set.

        Args:
            formula: The chemical formula of the desired entry.

        Returns:
            An interpolated GibbsComputedEntry object.
        """
        comp = Composition(formula).reduced_composition
        pd_entries = self.get_subset_in_chemsys([str(e) for e in comp.elements])

        energy = PhaseDiagram(pd_entries).get_hull_energy(comp) + tol

        adj = ConstantEnergyAdjustment(  # for keeping track of uncertainty
            value=0.0,
            uncertainty=0.05 * comp.num_atoms,  # conservative: 50 meV/atom uncertainty
            name="Interpolation adjustment",
            description="Keeps track of uncertainty in interpolation",
        )

        return ComputedEntry(
            comp, energy, energy_adjustments=[adj], entry_id="(Interpolated Entry!)"
        )

    @classmethod
    def from_pd(
        cls,
        pd: PhaseDiagram,
        temperature: float,
        include_nist_data=True,
        include_barin_data=False,
    ) -> "GibbsEntrySet":
        """
        Constructor method for building a GibbsEntrySet from an existing phase diagram.

        Args:
            pd: Phase Diagram object (pymatgen)
            temperature: Temperature [K] for determining Gibbs Free Energy of
                formation, dGf(T)
            include_nist_data: Whether to include NIST data in the entry set.
            include_barin_data: Whether to include Barin data in the entry set. Defaults
                to False. Warning: Barin data has not been verified. Use with caution.

        Returns:
            A GibbsEntrySet containing a collection of GibbsComputedEntry and
            experimental reference entry objects at the specified temperature.

        """
        gibbs_entries = []
        experimental_formulas = []

        if include_barin_data:
            warnings.warn(
                "##### WARNING ##### \n\n"
                "Barin experimental data was acquired through optical character"
                "recognition and has not been verified. Use at your own risk! \n\n"
                "##### WARNING #####"
            )

        for entry in pd.all_entries:
            experimental = False
            composition = entry.composition
            formula = composition.reduced_formula

            if (
                composition.is_element
                and entry not in pd.el_refs.values()
                or formula in experimental_formulas
            ):
                continue

            new_entries = []

            if include_nist_data and formula in NISTReferenceEntry.REFERENCES:
                try:
                    e = NISTReferenceEntry(
                        composition=Composition(formula), temperature=temperature
                    )
                    experimental = True
                    new_entries.append(e)
                except ValueError as error:
                    logging.warning(
                        f"Compound {formula} is in NIST-JANAF tables but at different temperatures!: {error}"
                    )
            if include_barin_data and formula in BarinReferenceEntry.REFERENCES:
                try:
                    e = BarinReferenceEntry(
                        composition=Composition(formula), temperature=temperature
                    )
                    experimental = True
                    new_entries.append(e)
                except ValueError as error:
                    logging.warning(
                        f"Compound {formula} is in Barin tables but not at this temperature! {error}"
                    )

            if experimental:
                experimental_formulas.append(formula)
            else:
                structure = entry.structure
                formation_energy_per_atom = pd.get_form_energy_per_atom(entry)

                new_entries.append(
                    GibbsComputedEntry.from_structure(
                        structure=structure,
                        formation_energy_per_atom=formation_energy_per_atom,
                        temperature=temperature,
                        energy_adjustments=None,
                        parameters=entry.parameters,
                        data=entry.data,
                        entry_id=entry.entry_id,
                    )
                )

            gibbs_entries.extend(new_entries)

        return cls(gibbs_entries)

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[ComputedStructureEntry],
        temperature: float,
        include_nist_data=True,
        include_barin_data=False,
    ) -> "GibbsEntrySet":
        """
        Constructor method for initializing GibbsEntrySet from T = 0 K
        ComputedStructureEntry objects, as acquired from a thermochemical
        database e.g. The Materials Project. Automatically expands the phase
        diagram for large chemical systems (10 or more elements) to avoid limitations
        of Qhull.

        Args:
            entries: List of ComputedStructureEntry objects, as downloaded from The
                Materials Project API.
            temperature: Temperature for estimating Gibbs free energy of formation [K]

        Returns:
            A GibbsEntrySet containing a collection of GibbsComputedEntry and
            experimental reference entry objects at the specified temperature.
        """
        e_set = EntrySet(entries)
        new_entries: Set[GibbsComputedEntry] = set()

        if len(e_set.chemsys) <= 9:  # Qhull algorithm struggles beyond 9 dimensions
            pd = PhaseDiagram(e_set)
            return cls.from_pd(
                pd,
                temperature,
                include_nist_data=include_nist_data,
                include_barin_data=include_barin_data,
            )

        pd_dict = expand_pd(list(e_set))
        for _, pd in tqdm(pd_dict.items()):
            gibbs_set = cls.from_pd(
                pd, temperature, include_nist_data, include_barin_data
            )
            new_entries.update(gibbs_set)

        return cls(list(new_entries))

    @property
    def entries_list(self) -> List[ComputedEntry]:
        """Returns a list of all entries in the entry set."""
        return list(sorted(self.entries, key=lambda e: e.composition.reduced_formula))

    @property
    def chemsys(self) -> set:
        """
        Returns:
            set representing the chemical system, e.g., {"Li", "Fe", "P", "O"}
        """
        chemsys = set()
        for e in self.entries:
            chemsys.update([el.symbol for el in e.composition.keys()])
        return chemsys

    def copy(self) -> "GibbsEntrySet":
        """Returns a copy of the entry set."""
        return GibbsEntrySet(entries=self.entries)

    def as_dict(self) -> dict:
        """
        Returns:
            JSON serializable dict representation of the entry set.
        """
        d = super().as_dict()
        d["entries"] = [e.as_dict() for e in self.entries]
        return d
