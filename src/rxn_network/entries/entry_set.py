from monty.json import MSONable
from typing import List, Optional, Union

from pymatgen.entries.entry_tools import EntrySet
from pymatgen.analysis.phase_diagram import PhaseDiagram

from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.entries.nist import NISTReferenceEntry


class GibbsEntrySet(EntrySet):
    def __init__(self, entries: List[Union[GibbsComputedEntry, NISTReferenceEntry]]):
        super().__init__(entries)

    @classmethod
    def from_pd(cls, pd: PhaseDiagram, temperature: float):
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
                volume_per_atom = structure.volume / structure.num_sites

                new_entry = GibbsComputedEntry(composition=composition,
                                               formation_energy_per_atom=formation_energy_per_atom,
                                               volume_per_atom=volume_per_atom,
                                               temperature=temperature,
                                               energy_adjustments=entry.energy_adjustments,
                                               parameters=entry.parameters,
                                               data=entry.data,
                                               entry_id=entry.entry_id,
                                               )

            gibbs_entries.append(new_entry)

        return cls(gibbs_entries)
