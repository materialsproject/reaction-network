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

    return OpenComputedReaction(entries=computed_rxn.entries,
                                coefficients=computed_rxn.coefficients,
                                chempots=chempots,
                                data=computed_rxn.data,
                                lowest_num_errors=computed_rxn.lowest_num_errors)


def test_energy(open_computed_rxn):
    expected = {"Na":{0: -2.70104842, -2: 5}}


def test_elements():
    pass


def test_copy():
    pass


def test_reverse():
    pass


def test_balance():
    pass


def test_total_chemical_system():
    pass