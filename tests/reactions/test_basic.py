"""
Test for BasicReaction. Several tests adapted from
test module for pymatgen.analysis.reaction_calculator
"""
import pytest
from pymatgen.core.composition import Composition, Element

from rxn_network.reactions.basic import BasicReaction


@pytest.fixture
def pre_balanced_rxn():
    """Returns a simple, pre-balanced iron oxidation reaction."""
    reactants = [Composition("Fe"), Composition("O2")]
    products = [Composition("Fe2O3")]
    coefficients = [-2, -1.5, 1]

    rxn = BasicReaction(
        compositions=reactants + products, coefficients=coefficients, balanced=True
    )
    return rxn


@pytest.fixture
def auto_balanced_rxn():
    """Returns the same iron oxidation reaction, after automatically balancing"""
    reactants = ["Fe", "O2"]
    products = ["Fe2O3"]
    return BasicReaction.from_formulas(reactants, products)


@pytest.mark.parametrize(
    "reactants, products, expected_rxn, expected_lowest_num_errors",
    [
        (["MgO"], ["MgO"], "MgO -> MgO", 0),
        (["Fe", "O2"], ["Fe2O3"], "2 Fe + 1.5 O2 -> Fe2O3", 0),
        (["FePO4", "LiPO3"], ["LiFeP2O7"], "FePO4 + LiPO3 -> LiFeP2O7", 0),
        (["Zn", "HCl"], ["ZnCl2", "H2"], "Zn + 2 HCl -> ZnCl2 + H2", 0),
        (
            ["LiCrO2", "La8Ti8O12", "O2"],
            ["LiLa3Ti3CrO12"],
            "LiCrO2 + 1.5 La2Ti2O3 + 2.75 O2 -> LiLa3Ti3CrO12",
            0,
        ),
        (["Na", "K2O"], ["Na2O", "K"], "2 Na + K2O -> Na2O + 2 K", 0),
        (
            ["NaMnO2", "YClO", "O2"],
            ["Y2Mn2O7", "NaCl"],
            "2 NaMnO2 + 2 YClO + 0.5 O2 -> Y2Mn2O7 + 2 NaCl",
            0,
        ),
        (
            ["La2O3", "Co2O3", "Li2ZrO3"],
            ["Li2O", "La2Zr2O7", "Li3CoO3"],
            "La2O3 + 0.3333 Co2O3 + 2 Li2ZrO3 -> Li2O + La2Zr2O7 + 0.6667 Li3CoO3",
            0,
        ),
        (["Li", "Cl", "Cl"], ["LiCl"], "Li + 0.25 Cl2 + 0.25 Cl2 -> LiCl", 0),
        (
            ["LiMnCl3", "LiCl", "MnCl2"],
            ["Li2MnCl4"],
            "LiMnCl3 + 3 LiCl + MnCl2 -> 2 Li2MnCl4",
            0,
        ),
        (["Fe", "O2"], ["Fe", "O2"], "Fe + O2 -> Fe + O2", 0),
        (
            ["Fe", "O2", "Na", "Li", "Cl"],
            ["FeO2", "NaCl", "Li2Cl2"],
            "Fe + O2 + Na + 2 Li + 1.5 Cl2 -> FeO2 + NaCl + 2 LiCl",
            0,
        ),
        (["FePO4", "O"], ["FePO4"], "FePO4 -> FePO4", 1),
        (
            ["La2O3", "Co2O3", "Li2ZrO3"],
            ["Li2O", "La2Zr2O7", "Li3CoO3", "Xe"],
            "La2O3 + 0.3333 Co2O3 + 2 Li2ZrO3 -> Li2O + La2Zr2O7 + 0.6667 Li3CoO3",
            1,
        ),
        (["FePO4", "Mn"], ["FePO4", "Xe"], "FePO4 -> FePO4", 2),
        (
            ["La2O3", "Co2O3", "Li2ZrO3"],
            ["Li2O", "La2Zr2O7", "Li3CoO3", "Xe", "XeNe"],
            "La2O3 + 0.3333 Co2O3 + 2 Li2ZrO3 -> Li2O + La2Zr2O7 + 0.6667 Li3CoO3",
            2,
        ),
        (
            ["LiCoO2"],
            ["La2O3", "Co2O3", "Li2O", "LiF", "CoF3"],
            "1.667 LiCoO2 + 0.3333 CoF3 -> Co2O3 + 0.3333 Li2O + LiF",
            2,
        ),
        (["LiCoO2", "Li2O"], ["ZrF4", "Co2O3"], "2 LiCoO2 -> Li2O + Co2O3", 2),
        (
            ["Fe", "Na", "Li2O", "Cl"],
            ["LiCl", "Na2O", "Xe", "FeCl", "Mn"],
            "Fe + Na + 0.5 Li2O + Cl2 -> LiCl + 0.5 Na2O + FeCl",
            2,
        ),
        (["XeMn", "Li"], ["S", "LiS2", "FeCl"], "Li + 2 S -> LiS2", 3),
    ],
)
def test_balance(reactants, products, expected_rxn, expected_lowest_num_errors):
    rxn = BasicReaction.from_formulas(reactants, products)
    assert str(rxn) == expected_rxn
    assert rxn.lowest_num_errors == expected_lowest_num_errors


def test_equality(pre_balanced_rxn, auto_balanced_rxn):
    assert pre_balanced_rxn == auto_balanced_rxn


@pytest.mark.parametrize(
    "reactants, products",
    [
        (["YMnO3"], ["YMnO3"]),
        (["YMnO3", "O2"], ["YMnO3", "O2"]),
        (["Li", "Na", "K"], ["Li", "Na", "K"]),
    ],
)
def test_is_identity(reactants, products):
    rxn = BasicReaction.from_formulas(reactants, products)

    assert rxn.is_identity is True


def test_get_el_amount(pre_balanced_rxn):
    rxn_normalized_oxygen = pre_balanced_rxn.copy().normalize_to_element(Element("O"))
    assert pre_balanced_rxn.get_el_amount(Element("Fe")) == 2
    assert pre_balanced_rxn.get_el_amount(Element("O")) == 3
    assert rxn_normalized_oxygen.get_el_amount(Element("O")) == 1


def test_copy(pre_balanced_rxn):
    rxn_copy = pre_balanced_rxn.copy()
    assert rxn_copy == pre_balanced_rxn.copy()


def test_equals():
    rxn1 = BasicReaction.from_formulas(["Y2O3", "MnO2"], ["Y2Mn2O7"])
    rxn2 = BasicReaction.from_formulas(["MnO2", "Y2O3"], ["Y2Mn2O7"])

    assert rxn1 == rxn2


def test_reverse():
    rxn = BasicReaction.from_formulas(
        ["La2O3", "Co2O3", "Li2ZrO3"], ["Li2O", "La2Zr2O7", "Li3CoO3", "Xe"]
    )
    rxn_reverse = rxn.reverse()

    assert (
        rxn_reverse
        == "Li2O + La2Zr2O7 + 0.6667 Li3CoO3 -> La2O3 + 0.3333 Co2O3 + 2 Li2ZrO3"
    )
    assert rxn_reverse.reverse() == rxn


def test_normalize(pre_balanced_rxn):
    rxn, factor = pre_balanced_rxn.normalized_repr_and_factor()
    rxn_o2_norm = pre_balanced_rxn.normalize_to(Composition("O2"))
    rxn_o2_norm_5 = pre_balanced_rxn.normalize_to(Composition("O2"), 5)

    assert rxn == "4 Fe + 3 O2 -> 2 Fe2O3"
    assert factor == 2
    assert rxn_o2_norm == "1.333 Fe + O2 -> 0.6667 Fe2O3"
    assert rxn_o2_norm_5 == "6.667 Fe + 5 O2 -> 3.333 Fe2O3"


def test_from_string(pre_balanced_rxn):
    prebalanced_rxn_from_str = BasicReaction.from_string("2 Fe + 1.5 O2 -> Fe2O3")

    rxn = BasicReaction.from_formulas(["Y2O3", "MnO2"], ["Y2Mn2O7"])
    rxn_from_str = BasicReaction.from_string("Y2O3 + 2 MnO2 -> Y2Mn2O7")

    assert pre_balanced_rxn == prebalanced_rxn_from_str
    assert rxn == rxn_from_str


def test_chemical_system(pre_balanced_rxn):
    complex_rxn = BasicReaction.from_formulas(
        ["LiCoO2", "Be", "Na"], ["La2O3", "Co2O3", "Li2O", "LiF", "CoF3"]
    )
    assert pre_balanced_rxn.chemical_system == "Fe-O"
    assert complex_rxn.chemical_system == "Be-Co-F-La-Li-Na-O"


def test_balanced():
    rxn_unbalanced = BasicReaction.from_formulas(["MnO2", "Y2O3"], ["YMn2O5"])
    rxn_balanced = BasicReaction.from_formulas(["YClO", "LiMnO2"], ["YMnO3", "LiCl"])
    rxn_balanced_2 = BasicReaction.from_formulas(
        ["Li", "Na2O", "FeCl2", "Y2O3"], ["YCl3", "Fe2O3", "NaCl", "Li2O"]
    )

    assert rxn_unbalanced.balanced is False
    assert rxn_balanced.balanced is True
    assert rxn_balanced_2.balanced is True


def test_get_coeff(pre_balanced_rxn):
    expected_coeffs = [-2, -1.5, 1]
    coeffs = [pre_balanced_rxn.get_coeff(c) for c in pre_balanced_rxn.compositions]

    assert coeffs == expected_coeffs


def test_energy(pre_balanced_rxn):
    with pytest.raises(ValueError) as error:
        assert pre_balanced_rxn.energy
    assert str(error.value) == "No energy for a basic reaction!"


def test_energy_per_atom(pre_balanced_rxn):
    with pytest.raises(ValueError) as error:
        assert pre_balanced_rxn.energy_per_atom
    assert str(error.value) == "No energy per atom for a basic reaction!"


def test_reactants_products(pre_balanced_rxn):
    assert pre_balanced_rxn.reactants == [Composition("Fe"), Composition("O2")]
    assert pre_balanced_rxn.products == [Composition("Fe2O3")]


def test_to_from_dict(pre_balanced_rxn):
    d = pre_balanced_rxn.as_dict()
    rxn = BasicReaction.from_dict(d)

    assert rxn == pre_balanced_rxn


def test_scientific_notation():
    reactants = ["FePO4"]
    products = ["FePO3.9999", "O2"]
    rxn = BasicReaction.from_formulas(reactants, products)
    rxn2 = BasicReaction.from_string("1e3 FePO4 + 20 CO -> 1e1 O2 + 1e3 Fe1P1O4 + 20 C")

    assert str(rxn) == "FePO4 -> Fe1P1O3.9999 + 5e-05 O2"
    assert str(rxn2) == "1000 FePO4 + 20 CO -> 10 O2 + 1000 FePO4 + 20 C"
