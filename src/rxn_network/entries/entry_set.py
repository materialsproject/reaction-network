from monty.json import MSONable
from typing import List, Optional, Union
from tqdm import tqdm

from pymatgen.entries.entry_tools import EntrySet
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.entries.nist import NISTReferenceEntry
from rxn_network.thermo.utils import expand_pd


class GibbsEntrySet(EntrySet):
    """
    An extension of pymatgen's EntrySet to include factory methods for constructing
    GibbsComputedEntry objects from zero-temperature ComputedStructureEntry objects.
    """
    def __init__(self, entries: List[Union[GibbsComputedEntry, NISTReferenceEntry]]):
        super().__init__(entries)

    def filter_by_stability(self, e_above_hull, include_polymorphs=False):
        """

        Args:
            e_above_hull:
            include_polymorphs:

        Returns:

        """
        pd_dict = expand_pd(self.entries)

        filtered_entries = set()
        all_comps = dict()

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

        return filtered_entries

    def get_min_entry_by_formula(self, formula):
        comp = Composition(formula).reduced_composition
        possible_entries = filter(
            lambda x: x.composition.reduced_composition == comp, self.entries
        )
        return sorted(possible_entries, key=lambda x: x.energy_per_atom)[0]

    def stabilize_entry(self, entry):
        pd = PhaseDiagram

    @classmethod
    def from_pd(cls, pd: PhaseDiagram, temperature: float) -> 'GibbsEntrySet':
        gibbs_entries = []
        for entry in pd.all_entries:
            if entry.composition.is_element and entry not in pd.el_refs.values():
                continue
            composition = entry.composition

            if composition.reduced_formula in NISTReferenceEntry.REFERENCES:
                new_entry = NISTReferenceEntry(composition=composition,
                                               temperature=temperature)
            else:
                structure = entry.structure
                formation_energy_per_atom = pd.get_form_energy_per_atom(entry)

                new_entry = GibbsComputedEntry.from_structure(structure=structure,
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
    def from_entries(cls, entries: List[ComputedStructureEntry], temperature: float) -> 'GibbsEntrySet':
        """
        Constructor method for initializing GibbsEntrySet from
        T = 0 K ComputedStructureEntry objects, as acquired from a thermochemical
        database e.g. The Materials Project.

        Args:
            entries ([ComputedStructureEntry]): List of ComputedStructureEntry objects,
                as downloaded from The Materials Project API.
            temp (float): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """

        e_set = EntrySet(entries)
        new_entries = set()
        if len(e_set.chemsys) <= 9:  # Qhull algorithm struggles beyond 9 dimensions
            pd = PhaseDiagram(e_set)
            return cls.from_pd(pd, temperature)

        pd_dict = expand_pd(list(e_set))
        for chemsys, pd in tqdm(pd_dict.items()):
            gibbs_set = cls.from_pd(pd, temperature)
            new_entries.update(gibbs_set)

        return cls(new_entries)
