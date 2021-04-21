""" Test for BasicReaction. Some tests adapted from pymatgen. """
import pytest

from pymatgen.core.composition import Composition
from rxn_network.reactions.basic import BasicReaction


@pytest.fixture
def prebalanced_rxn():
    """Returns a simple, pre-balanced iron oxidation reaction."""
    reactants = [Composition("Fe"), Composition("O2")]
    products = [Composition("Fe2O3")]
    coefficients = [-2, -1.5, 1]

    rxn = BasicReaction(reactants + products, coefficients, balanced=True)
    return rxn


def test_simple_balance():
    reactants = [Composition("Fe"), Composition("O2")]
    products = [Composition("Fe2O3")]
    rxn = BasicReaction.balance(reactants, products)

    assert str(rxn) == "2 Fe + 1.5 O2 -> Fe2O3"


def test_complex_balance():
    pass


def test_is_identity(prebalanced_rxn):
    rxn1 = BasicReaction.balance([Composition("YMnO3")], [Composition("YMnO3")])

    rxn2 = BasicReaction.balance([Composition("YMnO3"), Composition("O2")],
                                 [Composition("YMnO3"), Composition("O2")])

    assert rxn1.is_identity is True
    assert rxn2.is_identity is True
    assert prebalanced_rxn.is_identity is False


def test_copy():
    pass


def test_reverse():
    pass


def test_normalize(prebalanced_rxn):
    assert prebalanced_rxn.normalized_repr == "4 Fe + 3 O2 -> 2 Fe2O3"


def test_reduce():
    pass


def test_eq():
    pass


def test_from_str():
    pass
