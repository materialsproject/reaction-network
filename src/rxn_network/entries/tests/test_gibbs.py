" Tests for GibbsComputedEntry. Some tests adapted from pymatgen. "
import unittest
import json

from monty.json import MontyDecoder
from monty.serialization import loadfn

from pymatgen.core.composition import Composition

from rxn_network.entries.gibbs import GibbsComputedEntry
from rxn_network.utils import get_project_root


TEST_FILES_PATH = get_project_root() / "test_files"


class GibbsComputedStructureEntryTest(unittest.TestCase):
    def setUp(self):
        self.struct = loadfn(TEST_FILES_PATH / "structure_LiFe4P4O16.json")

        self.temps = [300, 600, 900, 1200, 1500, 1800]
        self.num_atoms = self.struct.composition.num_atoms
        self.entries_with_temps = {
            temp: GibbsComputedEntry.from_structure(
                structure=self.struct,
                formation_energy_per_atom=-2.436,
                temperature=temp,
                parameters=None,
                entry_id="Test LiFe4P4O16 structure",
            )
            for temp in self.temps
        }

        self.mp_entries = loadfn(TEST_FILES_PATH / "Mn-O_entries.json")

    def test_gf_sisso(self):
        energies = {
            300: -56.21273010866969,
            600: -51.52997063074788,
            900: -47.29888391585979,
            1200: -42.942338738866304,
            1500: -37.793417248809774,
            1800: -32.32513382051749,
        }
        for t in self.temps:
            self.assertAlmostEqual(self.entries_with_temps[t].energy, energies[t])

    def test_interpolation(self):
        temp = 450
        e = GibbsComputedEntry.from_structure(structure=self.struct,
                                              formation_energy_per_atom=-2.436,
                                              temperature=temp)
        self.assertAlmostEqual(e.energy, -53.7243542548528)

    def test_to_from_dict(self):
        test_entry = self.entries_with_temps[300]
        d = test_entry.as_dict()
        e = GibbsComputedEntry.from_dict(d)
        self.assertEqual(test_entry, e)
        self.assertAlmostEqual(e.energy, test_entry.energy)

    def test_str(self):
        self.assertIsNotNone(str(self.entries_with_temps[300]))

    def test_normalize(self):
        for e in self.entries_with_temps.values():
            normed_entry = e.normalize(mode="atom")
            self.assertAlmostEqual(e.uncorrected_energy, normed_entry.uncorrected_energy * self.num_atoms, 11)