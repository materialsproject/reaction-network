""" Tests for NISTReferenceEntry. """
import pytest
from pymatgen.core.periodic_table import Element
from pymatgen.entries.computed_entries import ManualEnergyAdjustment
from rxn_network.core import Composition
from rxn_network.entries.nist import NISTReferenceEntry


@pytest.fixture(scope="module")
def entries():
    comp = Composition("CO2")
    temps = [300, 600, 900, 1200, 1500, 1800]
    return {t: NISTReferenceEntry(composition=comp, temperature=t) for t in temps}


def test_invalid_formula():
    with pytest.raises(ValueError, match="AX not in reference data"):
        NISTReferenceEntry(Composition("AX"), temperature=300)


def test_invalid_temperature():
    with pytest.raises(ValueError, match="Temperature must be selected from range"):
        NISTReferenceEntry(Composition("K2CO3"), temperature=200)

    with pytest.raises(ValueError, match="Temperature must be selected from range"):
        NISTReferenceEntry(Composition("K2CO3"), temperature=2300)


def test_energy(entries):
    expected_energies = [
        -4.087606831162386,
        -4.095773877778095,
        -4.101640055931004,
        -4.105267551255239,
        -4.107236763002682,
        -4.107910440705754,
    ]
    actual_energies = [entry.energy for entry in entries.values()]

    assert actual_energies == pytest.approx(expected_energies)


def test_energy_per_atom(entries):
    expected_energies = [
        -1.362535610387462,
        -1.365257959259365,
        -1.3672133519770011,
        -1.3684225170850797,
        -1.369078921000894,
        -1.3693034802352513,
    ]
    actual_energies = [entry.energy_per_atom for entry in entries.values()]

    assert actual_energies == pytest.approx(expected_energies)


def test_correction_uncertainty(entries):
    assert all(e.correction_uncertainty == 0 for e in entries.values())


def test_correction_uncertainty_per_atom(entries):
    assert all(e.correction_uncertainty_per_atom == 0 for e in entries.values())


def test_is_experimental(entries):
    assert all(e.is_experimental for e in entries.values())


def test_is_element():
    assert not NISTReferenceEntry(Composition("CO2"), 300).is_element


def test_to_grand_entry(entries):
    entry = entries[300]

    chempots = {Element("O"): 0}
    grand_entry = entry.to_grand_entry(chempots)

    assert grand_entry.energy == pytest.approx(entry.energy)
    assert grand_entry.energy_per_atom != pytest.approx(entry.energy_per_atom)
    assert grand_entry.original_comp == entry.composition
    assert grand_entry.composition != entry.composition


def test_interpolate_energy(entries):
    comp = Composition("CO2")
    entry1 = NISTReferenceEntry(composition=comp, temperature=300)
    entry2 = NISTReferenceEntry(composition=comp, temperature=400)

    interpolated_entry = NISTReferenceEntry(composition=comp, temperature=350)
    assert interpolated_entry.energy == pytest.approx(0.5 * entry1.energy + 0.5 * entry2.energy)


def test_get_new_temperature(entries):
    entry = entries[300]

    new_temp = 400
    new_entry = entry.get_new_temperature(new_temp)

    assert new_entry.temperature == new_temp
    assert new_entry.energy_per_atom != pytest.approx(entry.energy_per_atom)


def test_as_dict(entries):
    d = entries[300].as_dict()

    assert type(d) == dict
    assert d.get("composition")
    assert d.get("temperature")


def test_from_dict(entries):
    e = entries[300]
    d = e.as_dict()
    assert NISTReferenceEntry.from_dict(d) == e


def test_equals(entries):
    comp = Composition("CO2")
    entry1 = NISTReferenceEntry(comp, temperature=300)
    entry2 = NISTReferenceEntry(
        composition=comp,
        temperature=300,
        energy_adjustments=[ManualEnergyAdjustment(0.1)],
    )

    assert entries[300] == entries[300]
    assert entries[300] != entries[600]

    assert entry1 != entry2


def test_unique_id(entries):
    entry = entries[300]
    assert str(entry.temperature) in entry.unique_id
    assert str(entry.composition.reduced_formula) in entry.unique_id
