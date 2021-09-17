""" Tests for ReactionSet."""
from pathlib import Path
import pytest

from monty.serialization import loadfn
from pymatgen.core.composition import Element
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet
from rxn_network.costs.softplus import Softplus

TEST_FILES_PATH = Path(__file__).parent.parent / "test_files"
RXNS_FILE = "ymno3_rxns.json.gz"


@pytest.fixture(scope="module")
def rxns():
    return loadfn(TEST_FILES_PATH / RXNS_FILE)


@pytest.fixture(scope="module")
def rxn_set(rxns):
    return ReactionSet.from_rxns(rxns)


def test_get_rxns(rxns, rxn_set):
    open_rxns = rxn_set.get_rxns(open_elem="O", chempot=0)
    assert rxn_set.get_rxns() == rxns
    assert open_rxns != rxns
    assert all([type(r) == OpenComputedReaction for r in open_rxns])
    assert all([r.chempots == {Element("O"): 0} for r in open_rxns])


def test_calculate_costs(rxns, rxn_set):
    cf = Softplus()
    assert rxn_set.calculate_costs(cf) == [cf.evaluate(r) for r in rxns]
