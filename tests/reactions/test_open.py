""" Tests for OpenComputedReaction. """
import pytest

from pymatgen.core.composition import Element
from rxn_network.reactions.open import OpenComputedReaction


@pytest.fixture(scope="module", params=["Na", "Mn", "O"])
def element(request):
    return request.param


@pytest.fixture(scope="module", params=[0, -2, -4, -6])
def chempot(request):
    return request.param


@pytest.fixture(scope="module")
def open_computed_rxn(computed_rxn, element, chempot):
    chempots = {Element(element): chempot}

    return OpenComputedReaction(
        entries=computed_rxn.entries,
        coefficients=computed_rxn.coefficients,
        chempots=chempots,
        data=computed_rxn.data,
        lowest_num_errors=computed_rxn.lowest_num_errors,
    )


def test_energy(open_computed_rxn):
    open_elem = list(open_computed_rxn.chempots.keys())[0]
    chempot = list(open_computed_rxn.chempots.values())[0]

    expected = {
        "Na": {
            0: -2.701048424999957,
            -2: -2.701048424999957,
            -4: -2.701048424999957,
            -6: -2.701048424999957,
        },
        "Mn": {
            0: -2.701048424999957,
            -2: -2.701048424999957,
            -4: -2.701048424999957,
            -6: -2.701048424999957,
        },
        "O": {
            0: -2.701048424999957,
            -2: -0.7010484249999713,
            -4: 1.2989515750000287,
            -6: 3.2989515750000145,
        },
    }

    assert open_computed_rxn.energy == pytest.approx(expected[open_elem.name][chempot])


def test_num_atoms(open_computed_rxn):
    open_elem = list(open_computed_rxn.chempots.keys())[0]

    expected = {"Na": 13, "Mn": 13, "O": 8}
    assert open_computed_rxn.num_atoms == pytest.approx(expected[open_elem.name])


def test_elements(open_computed_rxn):
    open_elem = list(open_computed_rxn.chempots.keys())[0]

    expected = {
        "Na": {Element(e) for e in ["Mn", "O", "Y", "Cl"]},
        "Mn": {Element(e) for e in ["Na", "O", "Y", "Cl"]},
        "O": {Element(e) for e in ["Na", "Mn", "Y", "Cl"]},
    }

    assert set(open_computed_rxn.elements) == expected[open_elem.name]


def test_copy(open_computed_rxn):
    assert open_computed_rxn.copy() == open_computed_rxn


def test_reverse(open_computed_rxn):
    open_computed_rxn_rev = open_computed_rxn.reverse()

    assert open_computed_rxn_rev.energy == -open_computed_rxn.energy
    assert open_computed_rxn_rev.reverse() == open_computed_rxn


def test_balance(open_computed_rxn):
    reactant_entries = open_computed_rxn.reactant_entries
    product_entries = open_computed_rxn.product_entries
    chempots = open_computed_rxn.chempots

    rxn = OpenComputedReaction.balance(reactant_entries, product_entries, chempots)

    assert open_computed_rxn == rxn


def test_total_chemical_system(open_computed_rxn):
    open_elem = list(open_computed_rxn.chempots.keys())[0].name

    elems = [e.name for e in open_computed_rxn.elements]
    elems.append(open_elem)

    total_chemsys = "-".join(sorted(set(elems)))
    assert open_computed_rxn.total_chemical_system == total_chemsys
