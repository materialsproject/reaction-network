""" Tests for NistReferenceEntry. """
import pytest

from pymatgen.core.composition import Composition

from rxn_network.entries.nist import NISTReferenceEntry


@pytest.fixture(scope="session")
def entries():
    comp = Composition("CO2")
    temps = [300, 600, 900, 1200, 1500, 1800]
    return {t: NISTReferenceEntry(composition=comp, temperature=t) for t in temps}


def test_invalid_formula():
    with pytest.raises(ValueError) as error:
        assert NISTReferenceEntry(Composition("AX"), temperature=300)
    assert str(error.value) == "Formula must be in NIST-JANAF thermochemical tables"


def test_invalid_temperature():
    with pytest.raises(ValueError) as error:
        assert NISTReferenceEntry(Composition("K2CO3"), temperature=200)
    assert str(error.value) == "Temperature must be selected from range: [300, 2000] K"

    with pytest.raises(ValueError) as error:
        assert NISTReferenceEntry(Composition("K2CO3"), temperature=2300)
    assert str(error.value) == "Temperature must be selected from range: [300, 2000] K"


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
    assert all([e.correction_uncertainty == 0 for e in entries.values()])


def test_correction_uncertainty_per_atom(entries):
    assert all([e.correction_uncertainty_per_atom == 0 for e in entries.values()])


def test_is_experimental(entries):
    assert all([e.is_experimental for e in entries.values()])


def test_as_dict(entries):
    d = entries[300].as_dict()

    assert type(d) == dict
    assert d.get("composition")
    assert d.get("temperature")


def test_from_dict(entries):
    e = entries[300]
    d = e.as_dict()
    assert NISTReferenceEntry.from_dict(d) == e
