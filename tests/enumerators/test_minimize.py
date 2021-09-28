""" Tests for MinimizeGibbsEnumerator and MinimizeGrandPotentialEnumerator """
import pytest
from pymatgen.core.composition import Element
from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator, MinimizeGrandPotentialEnumerator


@pytest.fixture
def gibbs_enumerator_default():
    return MinimizeGibbsEnumerator()


@pytest.fixture
def gibbs_enumerator_with_calculator():
    return MinimizeGibbsEnumerator(calculators=["ChempotDistanceCalculator"])


@pytest.fixture
def gibbs_enumerator_with_precursors():
    return MinimizeGibbsEnumerator(precursors=["Y2O3", "Mn2O3"])


@pytest.fixture
def gibbs_enumerator_with_target():
    return MinimizeGibbsEnumerator(target="YMnO3")


@pytest.fixture
def gibbs_enumerator_with_precursors_and_target():
    return MinimizeGibbsEnumerator(precursors=["Y2O3", "Mn2O3"], target="YMnO3")


@pytest.fixture
def grand_potential_enumerator():
    return MinimizeGrandPotentialEnumerator(open_elem=Element("O"), mu=0.0)


@pytest.fixture
def grand_potential_enumerator_with_precursors():
    return MinimizeGrandPotentialEnumerator(open_elem=Element("O"), mu=0.0,
                                            precursors=["Y2O3", "Mn2O3"])


@pytest.fixture
def grand_potential_enumerator_with_target():
    return MinimizeGrandPotentialEnumerator(open_elem=Element("O"),
                                            mu=0.0, target="Y2Mn2O7")


@pytest.fixture
def grand_potential_enumerator_with_precursors_and_target():
    return MinimizeGrandPotentialEnumerator(open_elem=Element("O"), mu=0.0,
                                            precursors=["Y2O3", "Mn2O3"],
                                            target="Y2Mn2O7")


def test_enumerate_gibbs(
    filtered_entries, gibbs_enumerator_default, gibbs_enumerator_with_calculator
):
    expected_num_rxns = 400

    for enumerator in [gibbs_enumerator_default, gibbs_enumerator_with_calculator]:
        rxns = enumerator.enumerate(filtered_entries)

        assert expected_num_rxns == len(rxns)
        assert len(rxns) == len(set(rxns))
        assert all([not r.is_identity for r in rxns])

        if enumerator.calculators:
            assert all([r.data["chempot_distance"] is not None for r in rxns])


def test_enumerate_gibbs_with_precursors(filtered_entries,
                                   basic_enumerator_with_precursors,
                                   basic_open_enumerator_with_precursors):

    for enumerator in [basic_enumerator_with_precursors,
                       basic_open_enumerator_with_precursors]:

        rxns = enumerator.enumerate(filtered_entries)
        precursors = enumerator.precursors

        for r in rxns:
            reactants = [i.reduced_formula for i in r.reactants]
            products = [i.reduced_formula for i in r.products]

            for precursor in precursors:
                assert precursor in reactants
                assert precursor not in products


def test_enumerate_gibbs_with_target(filtered_entries,
                               basic_enumerator_with_target,
                               basic_open_enumerator_with_target):

    for enumerator in [basic_enumerator_with_target,
                       basic_open_enumerator_with_target]:
        rxns = enumerator.enumerate(filtered_entries)
        target = enumerator.target

        for r in rxns:
            reactants = [i.reduced_formula for i in r.reactants]
            products = [i.reduced_formula for i in r.products]
            assert target not in reactants
            assert target in products


def test_enumerate_gibbs_with_precursors_and_target(
    filtered_entries, basic_enumerator_with_precursors_and_target
):
    rxns = basic_enumerator_with_precursors_and_target.enumerate(filtered_entries)

    assert len(rxns) == 1
    assert str(rxns[0]) == "Mn2O3 + Y2O3 -> 2 YMnO3"


def test_enumerate_grand(filtered_entries, grand_potential_enumerator):
    return True

