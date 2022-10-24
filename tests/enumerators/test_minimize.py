""" Tests for MinimizeGibbsEnumerator and MinimizeGrandPotentialEnumerator """
import pytest
from pymatgen.core.composition import Element

from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)


@pytest.fixture(scope="module")
def gibbs_enumerator_default():
    return MinimizeGibbsEnumerator(quiet=True)


@pytest.fixture(scope="module")
def gibbs_enumerator_with_precursors():
    return MinimizeGibbsEnumerator(precursors=["Y2O3", "Mn2O3"], quiet=True)


@pytest.fixture(scope="module")
def gibbs_enumerator_with_target():
    return MinimizeGibbsEnumerator(targets=["YMnO3"], quiet=True)


@pytest.fixture(scope="module")
def gibbs_enumerator_with_precursors_and_target():
    return MinimizeGibbsEnumerator(
        precursors=["Y2O3", "Mn2O3"], targets=["YMnO3"], quiet=True
    )


@pytest.fixture(scope="module")
def grand_potential_enumerator():
    return MinimizeGrandPotentialEnumerator(open_elem=Element("O"), mu=0.0, quiet=True)


@pytest.fixture(scope="module")
def grand_potential_enumerator_with_precursors():
    return MinimizeGrandPotentialEnumerator(
        open_elem=Element("O"), mu=0.0, precursors=["Y2O3", "Mn2O3"], quiet=True
    )


@pytest.fixture(scope="module")
def grand_potential_enumerator_with_target():
    return MinimizeGrandPotentialEnumerator(
        open_elem=Element("O"), mu=0.0, targets=["Y2Mn2O7"], quiet=True
    )


@pytest.fixture(scope="module")
def grand_potential_enumerator_with_precursors_and_target():
    return MinimizeGrandPotentialEnumerator(
        open_elem=Element("O"),
        mu=0.0,
        precursors=["Y2O3", "Mn2O3"],
        targets=["Y2Mn2O7"],
        quiet=True,
    )


def test_enumerate_gibbs(filtered_entries, gibbs_enumerator_default):
    expected_num_rxns = 109

    for enumerator in [gibbs_enumerator_default]:
        rxns = enumerator.enumerate(filtered_entries)

        assert len(rxns) == expected_num_rxns
        assert len(rxns) == len(set(rxns))
        assert all([not r.is_identity for r in rxns])


def test_enumerate_gibbs_with_precursors(
    filtered_entries, gibbs_enumerator_with_precursors
):
    expected_num_rxns = 2

    rxns = gibbs_enumerator_with_precursors.enumerate(filtered_entries)
    precursors = gibbs_enumerator_with_precursors.precursors

    assert len(rxns) == expected_num_rxns

    for r in rxns:
        reactants = [i.reduced_formula for i in r.reactants]
        products = [i.reduced_formula for i in r.products]

        for precursor in precursors:
            assert precursor in reactants
            assert precursor not in products


def test_enumerate_gibbs_with_target(filtered_entries, gibbs_enumerator_with_target):
    expected_num_rxns = 32

    rxns = gibbs_enumerator_with_target.enumerate(filtered_entries)
    targets = gibbs_enumerator_with_target.targets

    assert len(rxns) == expected_num_rxns

    for r in rxns:
        reactants = [i.reduced_formula for i in r.reactants]
        products = [i.reduced_formula for i in r.products]
        for target in targets:
            assert target not in reactants
            assert target in products


def test_enumerate_gibbs_with_precursors_and_target(
    filtered_entries, gibbs_enumerator_with_precursors_and_target
):
    rxns = gibbs_enumerator_with_precursors_and_target.enumerate(filtered_entries)

    assert {str(r) for r in rxns} == {"0.5 Mn2O3 + 0.5 Y2O3 -> YMnO3"}


def test_enumerate_grand_potential(filtered_entries, grand_potential_enumerator):
    expected_num_rxns = 42

    rxns = grand_potential_enumerator.enumerate(filtered_entries)

    assert len(rxns) == expected_num_rxns
    assert all([not r.is_identity for r in rxns])


def test_enumerate_grand_potential_precursors(
    filtered_entries, grand_potential_enumerator_with_precursors
):
    expected_rxns = {
        "Mn2O3 + 0.5 Y2O3 + 0.25 O2 -> YMn2O5",
        "Mn2O3 + Y2O3 + 0.5 O2 -> Y2Mn2O7",
        "0.5 Mn2O3 + 0.25 O2 -> MnO2",
    }

    rxns = grand_potential_enumerator_with_precursors.enumerate(filtered_entries)
    assert {str(r) for r in rxns} == expected_rxns


def test_enumerate_grand_potential_target(
    filtered_entries, grand_potential_enumerator_with_target
):
    expected_num_rxns = 14

    rxns = grand_potential_enumerator_with_target.enumerate(filtered_entries)
    targets = grand_potential_enumerator_with_target.targets

    assert len(rxns) == expected_num_rxns

    for r in rxns:
        reactants = [i.reduced_formula for i in r.reactants]
        products = [i.reduced_formula for i in r.products]
        for target in targets:
            assert target not in reactants
            assert target in products


def test_enumerate_grand_potential_precursors_target(
    filtered_entries, grand_potential_enumerator_with_precursors_and_target
):
    rxns = list(
        grand_potential_enumerator_with_precursors_and_target.enumerate(
            filtered_entries
        ).get_rxns()
    )

    assert len(rxns) == 1
    assert str(rxns[0]) == "Mn2O3 + Y2O3 + 0.5 O2 -> Y2Mn2O7"
