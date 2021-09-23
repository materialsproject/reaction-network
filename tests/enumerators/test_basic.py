""" Tests for BasicEnumerator and BasicOpenEnumerator """
from pathlib import Path
from monty.serialization import loadfn
import pytest

from rxn_network.enumerators.basic import BasicEnumerator, BasicOpenEnumerator

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"
RXNS_FILE = "ymno3_rxns.json.gz"


@pytest.fixture
def ymno3_rxns():
    return loadfn(TEST_FILES_PATH / RXNS_FILE)


@pytest.fixture
def filtered_entries(gibbs_entries):
    filtered_entries = gibbs_entries.filter_by_stability(0.0)
    return filtered_entries


@pytest.fixture
def basic_enumerator_default():
    return BasicEnumerator()


@pytest.fixture
def basic_enumerator_with_calculator():
    return BasicEnumerator(calculators=["ChempotDistanceCalculator"])


@pytest.fixture
def basic_enumerator_with_precursors():
    return BasicEnumerator(precursors=["Y2O3", "Mn2O3"])


@pytest.fixture
def basic_enumerator_with_target():
    return BasicEnumerator(target="YMnO3")


@pytest.fixture
def basic_enumerator_with_precursors_and_target():
    return BasicEnumerator(precursors=["Y2O3", "Mn2O3"], target="YMnO3")


@pytest.fixture
def basic_open_enumerator():
    return BasicOpenEnumerator(["O2"])


def basic_open_enumerator_with_target():
    return BasicOpenEnumerator(["O2"])



def test_enumerate(
    filtered_entries, basic_enumerator_default, basic_enumerator_with_calculator
):
    chemsys = "-".join(sorted(filtered_entries.chemsys))
    if chemsys != "Mn-O-Y":
        return

    expected_num_rxns = 538

    for enumerator in [basic_enumerator_default, basic_enumerator_with_calculator]:
        rxns = enumerator.enumerate(filtered_entries)

        assert expected_num_rxns == len(rxns)
        assert len(rxns) == len(set(rxns))
        assert all([not r.is_identity for r in rxns])

        if enumerator.calculators:
            assert all([r.data["chempot_distance"] is not None for r in rxns])


def test_enumerate_with_precursors(filtered_entries, basic_enumerator_with_precursors):
    chemsys = "-".join(sorted(filtered_entries.chemsys))
    if chemsys != "Mn-O-Y":
        return

    rxns = basic_enumerator_with_precursors.enumerate(filtered_entries)

    for r in rxns:
        reactants = [i.reduced_formula for i in r.reactants]
        products = [i.reduced_formula for i in r.products]
        assert "Y2O3" in reactants
        assert "Mn2O3" in reactants
        assert "Y2O3" not in products
        assert "Mn2O3" not in products


def test_enumerate_with_target(filtered_entries, basic_enumerator_with_target):
    chemsys = "-".join(sorted(filtered_entries.chemsys))
    if chemsys != "Mn-O-Y":
        return

    rxns = basic_enumerator_with_target.enumerate(filtered_entries)

    for r in rxns:
        reactants = [i.reduced_formula for i in r.reactants]
        products = [i.reduced_formula for i in r.products]
        assert "YMnO3" not in reactants
        assert "YMnO3" in products


def test_enumerate_with_precursors_and_target(
    filtered_entries, basic_enumerator_with_precursors_and_target
):
    chemsys = "-".join(sorted(filtered_entries.chemsys))
    if chemsys != "Mn-O-Y":
        return

    rxns = basic_enumerator_with_precursors_and_target.enumerate(filtered_entries)

    assert len(rxns) == 1
    assert str(rxns[0]) == "Mn2O3 + Y2O3 -> 2 YMnO3"
