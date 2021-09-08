""" Test for BasicReaction. Some tests adapted from pymatgen. """
import pytest
from pymatgen.core.composition import Composition

from rxn_network.reactions.basic import BasicReaction


@pytest.fixture
def user_balanced_rxn():
    """Returns a simple, pre-balanced iron oxidation reaction."""
    reactants = [Composition("Fe"), Composition("O2")]
    products = [Composition("Fe2O3")]
    coefficients = [-2, -1.5, 1]

    rxn = BasicReaction(reactants + products, coefficients, balanced=True)
    return rxn


@pytest.fixture
def auto_balanced_rxn():
    """Returns the same iron oxidation reaction, after automatically balancing"""
    reactants = [Composition("Fe"), Composition("O2")]
    products = [Composition("Fe2O3")]
    rxn = BasicReaction.balance(reactants, products)
    return rxn


@pytest.mark.parametrize(
    "reactants, products, expected_rxn",
    [
        (["Fe", "O2"], ["Fe2O3"], "2 Fe + 1.5 O2 -> Fe2O3"),
        (["Zn", "HCl"], ["ZnCl2", "H2"], "Zn + 2 HCl -> ZnCl2 + H2"),
        (["FePO4", "O"], ["FePO4"], "FePO4 -> FePO4"),
        (["LiCrO2", "La8Ti8O12", "O2"], ["LiLa3Ti3CrO12"],
         "LiCrO2 + 1.5 La2Ti2O3 + 2.75 O2 -> LiLa3Ti3CrO12"),


     ]
)
def test_balance(reactants, products, expected_rxn):
    reactants = [Composition(r) for r in reactants]
    products = [Composition(p) for p in products]
    rxn = BasicReaction.balance(reactants, products)
    assert rxn == expected_rxn


def test_equality(user_balanced_rxn, auto_balanced_rxn):
    assert user_balanced_rxn == auto_balanced_rxn


def test_is_identity(user_balanced_rxn):
    rxn1 = BasicReaction.balance([Composition("YMnO3")], [Composition("YMnO3")])

    rxn2 = BasicReaction.balance(
        [Composition("YMnO3"), Composition("O2")],
        [Composition("YMnO3"), Composition("O2")],
    )

    assert rxn1.is_identity is True
    assert rxn2.is_identity is True
    assert user_balanced_rxn.is_identity is False


def test_copy():
    pass


def test_reverse():
    pass


def test_normalize(user_balanced_rxn):
    assert user_balanced_rxn.normalized_repr == "4 Fe + 3 O2 -> 2 Fe2O3"


def test_reduce():
    pass


def test_from_str():
    pass
