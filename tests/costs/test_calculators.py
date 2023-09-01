""" Tests for ChempotDistanceCalculator """
from pathlib import Path

import pytest

from rxn_network.costs.calculators import (
    ChempotDistanceCalculator,
    PrimaryCompetitionCalculator,
    SecondaryCompetitionCalculator,
)
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"

cpd_expected_values = {
    "0.5 Y2O3 + 0.5 Mn2O3 -> YMnO3": {"sum": 0.48001, "max": 0.48001, "mean": 0.24},
    "2 YClO + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl": {
        "sum": 1.36979,
        "max": 1.36979,
        "mean": 0.19568,
    },
}


@pytest.fixture(
    params=[
        [["Y2O3", "Mn2O3"], ["YMnO3"]],
        [["YOCl", "NaMnO2", "O2"], ["Y2Mn2O7", "NaCl"]],
    ],
    scope="module",
)
def rxn(entries, request):
    reactants = request.param[0]
    products = request.param[1]
    reactant_entries = [entries.get_min_entry_by_formula(r) for r in reactants]
    product_entries = [entries.get_min_entry_by_formula(p) for p in products]
    return ComputedReaction.balance(reactant_entries, product_entries)


@pytest.fixture
def rxns(entries):
    return [
        ComputedReaction.balance(
            [entries.get_min_entry_by_formula(r) for r in reactants],
            [entries.get_min_entry_by_formula(p) for p in products],
        )
        for reactants, products in [
            (["Y2O3", "Mn2O3"], ["YMnO3"]),
            (["YOCl", "NaMnO2", "O2"], ["Y2Mn2O7", "NaCl"]),
        ]
    ]


@pytest.fixture(params=["sum", "max", "mean"], scope="module")
def mu_func(request):
    return request.param


@pytest.fixture
def cpd(entries):
    return ChemicalPotentialDiagram(entries)


@pytest.fixture
def cpd_calculator(cpd, mu_func):
    return ChempotDistanceCalculator(cpd=cpd, mu_func=mu_func)


@pytest.fixture
def primary_competition_calculator(irh_batio):
    return PrimaryCompetitionCalculator(irh_batio)


@pytest.fixture
def secondary_competition_calculator(irh_batio):
    return SecondaryCompetitionCalculator(irh_batio)


@pytest.fixture
def stable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 2 BaO -> Ba2TiO4":
            return r


@pytest.fixture
def unstable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 0.9 BaO -> 0.1 Ti10O11 + 0.9 BaO2":
            return r


def test_cpd_calculate(cpd_calculator, rxn):
    actual_cost = cpd_calculator.calculate(rxn)
    expected_cost = cpd_expected_values[str(rxn)][cpd_calculator.mu_func.__name__]

    assert actual_cost == pytest.approx(expected_cost)


def test_cpd_decorate(cpd_calculator, rxn):
    rxn_missing_data = rxn.copy()
    rxn_missing_data.data = None

    rxn_dec = cpd_calculator.decorate(rxn)
    rxn_missing_data_dec = cpd_calculator.decorate(rxn_missing_data)

    actual_cost = rxn_dec.data[cpd_calculator.name]
    expected_cost = cpd_expected_values[str(rxn)][cpd_calculator.mu_func.__name__]

    assert type(rxn_dec) == ComputedReaction
    assert actual_cost == pytest.approx(expected_cost)
    assert rxn_missing_data_dec.data[cpd_calculator.name] == actual_cost


def test_cpd_calculate_many(cpd_calculator, rxns):
    actual_costs = cpd_calculator.calculate_many(rxns)
    expected_costs = [cpd_calculator.calculate(rxn) for rxn in rxns]

    assert actual_costs == pytest.approx(expected_costs)


def test_cpd_decorate_many(cpd_calculator, rxns):
    rxns_missing_data = [rxn.copy() for rxn in rxns]
    for rxn in rxns_missing_data:
        rxn.data = None

    rxns_dec = cpd_calculator.decorate_many(rxns)
    rxns_missing_data_dec = cpd_calculator.decorate_many(rxns_missing_data)

    actual_costs = [rxn.data[cpd_calculator.name] for rxn in rxns_dec]
    expected_costs = [cpd_calculator.calculate(rxn) for rxn in rxns]

    assert type(rxns_dec) == list
    assert actual_costs == pytest.approx(expected_costs)
    assert rxns_missing_data_dec == rxns_dec


def test_cpd_calculator_from_entries(entries, mu_func, rxn):
    calc = ChempotDistanceCalculator.from_entries(entries=entries, mu_func=mu_func)

    actual_cost = calc.calculate(rxn)
    expected_cost = cpd_expected_values[str(rxn)][calc.mu_func.__name__]

    assert type(calc) == ChempotDistanceCalculator
    assert actual_cost == pytest.approx(expected_cost)


def test_primary_competition_calculate(
    primary_competition_calculator, stable_rxn, unstable_rxn, irh_batio
):
    assert primary_competition_calculator.calculate(stable_rxn) == pytest.approx(
        irh_batio.get_primary_competition(stable_rxn)
    )
    assert primary_competition_calculator.calculate(unstable_rxn) == pytest.approx(
        irh_batio.get_primary_competition(unstable_rxn)
    )


def test_secondary_competition_calculate(
    secondary_competition_calculator, stable_rxn, unstable_rxn, irh_batio
):
    assert secondary_competition_calculator.calculate(stable_rxn) == pytest.approx(
        irh_batio.get_secondary_competition(stable_rxn)
    )
    assert secondary_competition_calculator.calculate(unstable_rxn) == pytest.approx(
        irh_batio.get_secondary_competition(unstable_rxn)
    )
