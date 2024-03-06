"""Tests for BasicEnumerator and BasicOpenEnumerator"""

from pathlib import Path

import pytest
from monty.serialization import loadfn
from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"
RXNS_FILE = "ymno3_rxns.json.gz"


@pytest.fixture(scope="module")
def ymno3_rxns():
    return loadfn(TEST_FILES_PATH / RXNS_FILE)


@pytest.fixture(scope="module")
def basic_enumerator_default():
    return BasicEnumerator(quiet=True)


@pytest.fixture(scope="module")
def basic_enumerator_more_constraints():
    return BasicEnumerator(quiet=True, max_num_constraints=2)


@pytest.fixture(scope="module")
def basic_enumerator_with_precursors():
    return BasicEnumerator(precursors=["Y2O3", "Mn2O3"], quiet=True)


@pytest.fixture(scope="module")
def basic_enumerator_with_target():
    return BasicEnumerator(targets=["YMnO3"], quiet=True)


@pytest.fixture(scope="module")
def basic_enumerator_with_precursors_and_target():
    return BasicEnumerator(precursors=["Y2O3", "Mn2O3"], targets=["YMnO3"], quiet=True)


@pytest.fixture(scope="module")
def basic_open_enumerator():
    return BasicOpenEnumerator(["O2"], quiet=True)


@pytest.fixture(scope="module")
def basic_open_enumerator_with_precursors():
    return BasicOpenEnumerator(["O2"], precursors=["Y2O3", "Mn2O3"], quiet=True)


@pytest.fixture(scope="module")
def basic_open_enumerator_with_target():
    return BasicOpenEnumerator(["O2"], targets=["Y2Mn2O7"], quiet=True)


@pytest.fixture(scope="module")
def basic_open_enumerator_with_precursors_and_target():
    return BasicOpenEnumerator(
        ["O2"],
        precursors=["Y2O3", "Mn2O3"],
        targets=["Y2Mn2O7"],
        quiet=True,
    )


def test_enumerate(filtered_entries, basic_enumerator_default, basic_enumerator_more_constraints):
    expected_num_rxns_1 = 496
    expected_num_rxns_2 = 538

    for enumerator, expected_num_rxns in zip(
        [basic_enumerator_default, basic_enumerator_more_constraints],
        [expected_num_rxns_1, expected_num_rxns_2],
    ):
        rxns = enumerator.enumerate(filtered_entries)

        assert expected_num_rxns == len(rxns)
        assert len(rxns) == len(set(rxns))  # no duplicates
        assert all(not r.is_identity for r in rxns)


def test_enumerate_with_precursors(
    filtered_entries,
    basic_enumerator_with_precursors,
    basic_open_enumerator_with_precursors,
):
    for enumerator in [
        basic_enumerator_with_precursors,
        basic_open_enumerator_with_precursors,
    ]:
        precursors = set(enumerator.precursors)
        rxns = enumerator.enumerate(filtered_entries)

        for r in rxns:
            reactants = {i.reduced_formula for i in r.reactants}
            assert precursors & reactants


def test_enumerate_with_target(filtered_entries, basic_enumerator_with_target, basic_open_enumerator_with_target):
    for enumerator in [basic_enumerator_with_target, basic_open_enumerator_with_target]:
        rxns = enumerator.enumerate(filtered_entries)
        targets = enumerator.targets

        for r in rxns:
            reactants = [i.reduced_formula for i in r.reactants]
            products = [i.reduced_formula for i in r.products]

            for target in targets:
                assert target not in reactants
                assert target in products


def test_enumerate_with_precursors_and_target(filtered_entries, basic_enumerator_with_precursors_and_target):
    rxns = list(basic_enumerator_with_precursors_and_target.enumerate(filtered_entries).get_rxns())

    assert len(rxns) == 1
    rxn_str = str(rxns[0])
    assert rxn_str == "Mn2O3 + Y2O3 -> 2 YMnO3" or rxn_str == "Y2O3 + Mn2O3 -> 2 YMnO3"


def test_open_enumerate_with_precursors_and_target(filtered_entries, basic_open_enumerator_with_precursors_and_target):
    rxns = list(basic_open_enumerator_with_precursors_and_target.enumerate(filtered_entries).get_rxns())

    assert len(rxns) == 1
    assert {c.reduced_formula for c in rxns[0].reactants} == {"Y2O3", "Mn2O3", "O2"}
    assert {c.reduced_formula for c in rxns[0].products} == {"Y2Mn2O7"}


def test_open_enumerate(filtered_entries, basic_open_enumerator):
    expected_num_rxns = 168

    rxns = list(basic_open_enumerator.enumerate(filtered_entries).get_rxns())

    assert expected_num_rxns == len(rxns)
    assert len(rxns) == len(set(rxns))
    assert all(not r.is_identity for r in rxns)
