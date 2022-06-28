""" Tests for ReactionSet."""
from pathlib import Path
import pytest

from monty.serialization import loadfn
from rxn_network.core.composition import Composition
from pymatgen.core.composition import Element
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.costs.softplus import Softplus


@pytest.fixture(scope="module")
def rxn_set(ymno_rxns):
    return ReactionSet.from_rxns(ymno_rxns)


@pytest.fixture(scope="module")
def open_rxn_set(ymno_rxns):
    return ReactionSet.from_rxns(ymno_rxns, open_elem="O", chempot=0)


def test_get_rxns(ymno_rxns, rxn_set, open_rxn_set):
    open_rxns = open_rxn_set.get_rxns()
    assert rxn_set.get_rxns() == ymno_rxns
    assert open_rxns != ymno_rxns
    assert all([type(r) == OpenComputedReaction for r in open_rxns])
    assert all([r.chempots == {Element("O"): 0} for r in open_rxns])


def test_calculate_costs(ymno_rxns, rxn_set):
    cf = Softplus()
    assert rxn_set.calculate_costs(cf) == [cf.evaluate(r) for r in ymno_rxns]
