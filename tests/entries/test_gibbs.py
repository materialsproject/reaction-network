"""Tests for GibbsComputedEntry. Some tests adapted from pymatgen."""

from pathlib import Path

import pytest
from monty.serialization import loadfn
from pymatgen.core.composition import Element
from rxn_network.entries.gibbs import GibbsComputedEntry

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"


@pytest.fixture()
def structure():
    return loadfn(TEST_FILES_PATH / "structure_LiFe4P4O16.json")


@pytest.fixture()
def entry(structure):
    return GibbsComputedEntry.from_structure(
        structure=structure,
        formation_energy_per_atom=-2.436,
        temperature=300,
        parameters=None,
        entry_id="LiFe4P4O16 test structure",
    )


@pytest.fixture()
def entries_temps_dict(structure):
    struct = structure

    temps = [300, 600, 900, 1200, 1500, 1800]
    return {
        temp: GibbsComputedEntry.from_structure(
            structure=struct,
            formation_energy_per_atom=-2.436,
            temperature=temp,
            parameters=None,
            entry_id="Test LiFe4P4O16 structure",
        )
        for temp in temps
    }


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
    e = GibbsComputedEntry.from_structure(structure=structure, formation_energy_per_atom=-2.436, temperature=temp)
    assert e.energy == pytest.approx(-53.7243542548528)


def test_to_from_dict(entry):
    d = entry.as_dict()
    e = GibbsComputedEntry.from_dict(d)
    assert e == entry
    assert e.energy == pytest.approx(entry.energy)


def test_get_new_temperature(entry):
    new_temp = 600  # != original temp of 450
    new_entry = entry.get_new_temperature(new_temp)

    assert new_entry.temperature == new_temp
    assert new_entry.formation_energy_per_atom == pytest.approx(entry.formation_energy_per_atom)
    assert new_entry.energy != entry.energy


def test_to_grand_entry(entry):
    chempots = {Element("O"): 0}
    grand_entry = entry.to_grand_entry(chempots)

    assert grand_entry.energy == pytest.approx(entry.energy)
    assert grand_entry.energy_per_atom != pytest.approx(entry.energy_per_atom)
    assert grand_entry.original_comp == entry.composition
    assert grand_entry.composition != entry.composition


def test_str(entry):
    assert str(entry) is not None


def test_normalize(entries_temps_dict):
    num_atoms = 25
    for e in entries_temps_dict.values():
        normed_entry = e.normalize(mode="atom")
        assert e.uncorrected_energy == pytest.approx(normed_entry.uncorrected_energy * num_atoms)


def test_is_experimental(entry):
    assert not entry.is_experimental

    entry2 = entry.get_new_temperature(300)
    entry2.data["theoretical"] = False

    entry3 = entry.get_new_temperature(300)
    entry3.data["icsd_ids"] = ["123456"]

    assert entry2.is_experimental
    assert entry3.is_experimental


def test_eq(entry):
    entry2 = entry.get_new_temperature(600)
    assert entry != entry2

    entry3 = entry.get_new_temperature(300)
    assert entry == entry3


def test_eq_different_class(entry, mp_entries):
    assert entry != mp_entries[0]


def test_eq_no_entry_id(entry):
    e1 = entry.copy()
    e2 = entry.copy()

    e1.entry_id = None
    e2.entry_id = None

    assert e1 == e2


def test_invalid_temperature(entry):
    with pytest.raises(ValueError, match="Temperature must be selected from range"):
        entry.get_new_temperature(299)
    with pytest.raises(ValueError, match="Temperature must be selected from range"):
        entry.get_new_temperature(2001)


def test_unique_id(entry):
    assert str(entry.temperature) in entry.unique_id
    assert str(entry.entry_id) in entry.unique_id
