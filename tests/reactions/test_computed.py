""" Tests for ComputedReaction. Some tests adapted from pymatgen. """
import pytest
from pathlib import Path
from monty.serialization import loadfn

from rxn_network.reactions.computed import ComputedReaction
from rxn_network.entries.gibbs import GibbsComputedEntry

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"
ENTRIES_FILE = "yocl_namno2_rxn_entries.json.gz"


@pytest.fixture(scope="session")
def entries():
    return loadfn(TEST_FILES_PATH / ENTRIES_FILE)


@pytest.fixture(scope="session")
def reactants(entries):
    return [entries["YOCl"], entries["NaMnO2"], entries["O2"]]


@pytest.fixture(scope="session")
def products(entries):
    return [entries["Y2Mn2O7"], entries["NaCl"]]


@pytest.fixture(scope="session")
def pre_balanced_rxn(reactants, products):
    """Returns a simple, pre-balanced computed reaction."""
    coefficients = [-2, -2, -0.5, 1, 2]
    rxn = ComputedReaction(
        entries=reactants + products,
        coefficients=coefficients,
    )
    return rxn


@pytest.fixture(scope="session")
def auto_balanced_rxn(reactants, products):
    """Returns the same iron oxidation reaction, after automatically balancing"""
    return ComputedReaction.balance(
        reactant_entries=reactants, product_entries=products
    )


@pytest.fixture(scope="session")
def gibbs_balanced_rxn(gibbs_entries):
    """Returns a simple, pre-balanced computed reaction using GibbsComputedentry objects."""
    return ComputedReaction.balance(
        reactant_entries=[
            gibbs_entries.get_min_entry_by_formula("Y2O3"),
            gibbs_entries.get_min_entry_by_formula("Mn2O3"),
        ],
        product_entries=[gibbs_entries.get_min_entry_by_formula("YMnO3")],
    )


def test_energy(pre_balanced_rxn, auto_balanced_rxn):
    expected_energy = -2.701048

    assert pre_balanced_rxn.energy == pytest.approx(expected_energy)
    assert auto_balanced_rxn.energy == pytest.approx(expected_energy)


def test_energy_per_atom(pre_balanced_rxn, auto_balanced_rxn):
    expected_energy_per_atom = -0.1800700

    assert pre_balanced_rxn.energy_per_atom == pytest.approx(expected_energy_per_atom)
    assert auto_balanced_rxn.energy_per_atom == pytest.approx(expected_energy_per_atom)


def test_energy_uncertainty(pre_balanced_rxn, auto_balanced_rxn):
    expected_energy_uncertainty = 0.0229486383

    assert pre_balanced_rxn.energy_uncertainty == pytest.approx(
        expected_energy_uncertainty
    )
    assert auto_balanced_rxn.energy_uncertainty == pytest.approx(
        expected_energy_uncertainty
    )


def test_energy_uncertainty_per_atom(pre_balanced_rxn, auto_balanced_rxn):
    expected_energy_uncertainty_per_atom = 0.0015299092

    assert pre_balanced_rxn.energy_uncertainty_per_atom == pytest.approx(
        expected_energy_uncertainty_per_atom
    )
    assert auto_balanced_rxn.energy_uncertainty_per_atom == pytest.approx(
        expected_energy_uncertainty_per_atom
    )


def test_copy(pre_balanced_rxn, auto_balanced_rxn):
    pre_balanced_rxn_copy = pre_balanced_rxn.copy()
    auto_balanced_rxn_copy = auto_balanced_rxn.copy()

    assert (
        pre_balanced_rxn
        == auto_balanced_rxn
        == pre_balanced_rxn_copy
        == auto_balanced_rxn_copy
    )


def test_reverse(pre_balanced_rxn):
    pre_balanced_rxn_rev = pre_balanced_rxn.reverse()

    assert pre_balanced_rxn.energy == -pre_balanced_rxn_rev.energy

    assert pre_balanced_rxn == pre_balanced_rxn_rev.reverse()


def test_get_new_temperature(pre_balanced_rxn, gibbs_balanced_rxn):
    with pytest.raises(AttributeError):
        new_rxn = pre_balanced_rxn.get_new_temperature(
            1500
        )  # this reaction only uses ComputedStructureEntry

    new_rxn = gibbs_balanced_rxn.get_new_temperature(1500)
    for e in new_rxn.entries:
        assert e.temperature == 1500
