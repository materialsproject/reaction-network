"""
An entry set class for acquiring entries with Gibbs formation energies
"""
from typing import List, Optional, Union, Set, Dict

from monty.json import MontyDecoder
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,
    ConstantEnergyAdjustment,
)
from pymatgen.entries.entry_tools import EntrySet
from tqdm.auto import tqdm

from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.entries.nist import NISTReferenceEntry
from rxn_network.thermo.utils import expand_pd


class GibbsEntrySet(EntrySet):
    """
    An extension of pymatgen's EntrySet to include factory methods for constructing
    GibbsComputedEntry objects from zero-temperature ComputedStructureEntry objects.
    """

    def __init__(self, entries: List[Union[GibbsComputedEntry, NISTReferenceEntry]]):
        """
        The supplied collection of entries will automatically be converted to a set of
        unique entries.

        Args:
            entries: A collection of entry objects that will make up the entry set.
        """
        super().__init__(entries)
        self.entries_list = list(
            sorted(entries, key=lambda e: e.composition.reduced_formula)
        )
        self.build_indices()

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
        all_comps: Dict[str, Union[GibbsComputedEntry, NISTReferenceEntry]] = dict()

        for chemsys, pd in pd_dict.items():
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

    def stabilize_entry(self, entry: ComputedEntry, tol: float = 1e-6) -> ComputedEntry:
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

    @classmethod
    def from_pd(cls, pd: PhaseDiagram, temperature: float) -> "GibbsEntrySet":
        """
        Constructor method for building a GibbsEntrySet from an existing phase diagram.

        Args:
            pd: Phase Diagram object (pymatgen)
            temperature: Temperature [K] for determining Gibbs Free Energy of
                formation, dGf(T)

        Returns:
            A GibbsEntrySet containing a collection of GibbsComputedEntry and
            experimental reference entry objects at the specified temperature.

        """
        gibbs_entries = []
        for entry in pd.all_entries:
            if entry.composition.is_element and entry not in pd.el_refs.values():
                continue
            composition = entry.composition

            if composition.reduced_formula in NISTReferenceEntry.REFERENCES:
                new_entry = NISTReferenceEntry(
                    composition=composition, temperature=temperature
                )
            else:
                structure = entry.structure
                formation_energy_per_atom = pd.get_form_energy_per_atom(entry)

                new_entry = GibbsComputedEntry.from_structure(
                    structure=structure,
                    formation_energy_per_atom=formation_energy_per_atom,
                    temperature=temperature,
                    energy_adjustments=None,
                    parameters=entry.parameters,
                    data=entry.data,
                    entry_id=entry.entry_id,
                )

            gibbs_entries.append(new_entry)

        return cls(gibbs_entries)

    @classmethod
    def from_entries(
        cls, entries: List[ComputedStructureEntry], temperature: float
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
            return cls.from_pd(pd, temperature)

        pd_dict = expand_pd(list(e_set))
        for chemsys, pd in tqdm(pd_dict.items()):
            gibbs_set = cls.from_pd(pd, temperature)
            new_entries.update(gibbs_set)

        return cls(list(new_entries))

    def copy(self) -> "GibbsEntrySet":
        return GibbsEntrySet(entries=self.entries)
