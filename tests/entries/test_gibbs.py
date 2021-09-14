""" Tests for GibbsComputedEntry. Some tests adapted from pymatgen."""
from pathlib import Path

import pytest
from monty.serialization import loadfn

from rxn_network.entries.gibbs import GibbsComputedEntry

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"


@pytest.fixture
def structure():
    struct = loadfn(TEST_FILES_PATH / "structure_LiFe4P4O16.json")
    return struct


@pytest.fixture
def entry(structure):
    entry = GibbsComputedEntry.from_structure(
        structure=structure,
        formation_energy_per_atom=-2.436,
        temperature=300,
        parameters=None,
        entry_id="LiFe4P4O16 test structure",
    )
    return entry


@pytest.fixture
def entries_temps_dict(structure):
    struct = structure

    temps = [300, 600, 900, 1200, 1500, 1800]
    entries_with_temps = {
        temp: GibbsComputedEntry.from_structure(
            structure=struct,
            formation_energy_per_atom=-2.436,
            temperature=temp,
            parameters=None,
            entry_id="Test LiFe4P4O16 structure",
        )
        for temp in temps
    }
    return entries_with_temps


def test_gf_sisso(entries_temps_dict):
    test_energies = {
        300: -56.21273010866969,
        600: -51.52997063074788,
        900: -47.29888391585979,
        1200: -42.942338738866304,
        1500: -37.793417248809774,
        1800: -32.32513382051749,
    }
    entry_energies = {t: e.energy for t, e in entries_temps_dict.items()}

    assert entry_energies == pytest.approx(test_energies)


def test_interpolation(structure):
    temp = 450
    e = GibbsComputedEntry.from_structure(
        structure=structure, formation_energy_per_atom=-2.436, temperature=temp
    )
    assert e.energy == pytest.approx(-53.7243542548528)


def test_to_from_dict(entry):
    d = entry.as_dict()
    e = GibbsComputedEntry.from_dict(d)
    assert e == entry
    assert e.energy == pytest.approx(entry.energy)


def test_str(entry):
    assert str(entry) is not None


def test_normalize(entries_temps_dict):
    num_atoms = 25
    for e in entries_temps_dict.values():
        normed_entry = e.normalize(mode="atom")
        assert e.uncorrected_energy == pytest.approx(
            normed_entry.uncorrected_energy * num_atoms
        )
