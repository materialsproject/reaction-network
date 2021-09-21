""" Tests for ChempotDistanceCalculator """
from pathlib import Path

import pytest
from monty.serialization import loadfn

import numpy as np

from rxn_network.costs.calculators import ChempotDistanceCalculator
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.computed import ComputedReaction

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"

answers = {
    "0.5 Y2O3 + 0.5 Mn2O3 -> YMnO3": {
        np.sum: 0.480008216,
        np.max: 0.480008216,
        np.mean: 0.240004108,
    },
    "2 YClO + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl": {
        np.sum: 1.369790046,
        np.max: 1.369790045,
        np.mean: 0.195684292,
    },
}


@pytest.fixture(params=["Cl-Mn-Na-O-Y_entries.json.gz"])
def entries(request):
    entries = loadfn(TEST_FILES_PATH / request.param)
    return GibbsEntrySet(entries)


@pytest.fixture(
    params=[
        [["Y2O3", "Mn2O3"], ["YMnO3"]],
        [["YOCl", "NaMnO2", "O2"], ["Y2Mn2O7", "NaCl"]],
    ]
)
def rxn(entries, request):
    reactants = request.param[0]
    products = request.param[1]
    reactant_entries = [entries.get_min_entry_by_formula(r) for r in reactants]
    product_entries = [entries.get_min_entry_by_formula(p) for p in products]
    return ComputedReaction.balance(reactant_entries, product_entries)


@pytest.fixture(params=["sum", "max", "mean"])
def mu_func(request):
    return request.param


@pytest.fixture
def cpd(entries):
    return ChemicalPotentialDiagram(entries)


@pytest.fixture
def calculator(cpd, mu_func):
    return ChempotDistanceCalculator(cpd=cpd, mu_func=mu_func)


def test_calculate(calculator, rxn):
    actual_cost = calculator.calculate(rxn)
    expected_cost = answers[str(rxn)][calculator.mu_func]

    assert actual_cost == pytest.approx(expected_cost)


def test_decorate(calculator, rxn):
    rxn_dec = calculator.decorate(rxn)

    actual_cost = rxn_dec.data[calculator.name]
    expected_cost = answers[str(rxn)][calculator.mu_func]

    assert type(rxn_dec) == ComputedReaction
    assert pytest.approx(expected_cost, actual_cost)


def test_from_entries(entries, mu_func, rxn):
    calc = ChempotDistanceCalculator.from_entries(entries=entries, mu_func=mu_func)

    actual_cost = calc.calculate(rxn)
    expected_cost = answers[str(rxn)][calc.mu_func]

    assert type(calc) == ChempotDistanceCalculator
    assert pytest.approx(expected_cost, actual_cost)
