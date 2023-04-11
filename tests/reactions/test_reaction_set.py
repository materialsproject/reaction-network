""" Tests for ReactionSet."""
from pathlib import Path

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.costs.softplus import Softplus
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet


@pytest.fixture(scope="module")
def rxn_set(ymno3_rxns):
    return ReactionSet.from_rxns(ymno3_rxns)


@pytest.fixture(scope="module")
def open_rxn_set(ymno3_rxns):
    return ReactionSet.from_rxns(ymno3_rxns, open_elem="O", chempot=0)


def test_get_rxns(ymno3_rxns, rxn_set, open_rxn_set):
    ymno3_rxns_set = set(
        ymno3_rxns
    )  # order may change when creating ReactionSet object
    open_rxns = set(open_rxn_set.get_rxns())
    assert set(rxn_set.get_rxns()) == ymno3_rxns_set
    assert open_rxns != ymno3_rxns_set
    assert all([type(r) == OpenComputedReaction for r in open_rxns])
    assert all([r.chempots == {Element("O"): 0} for r in open_rxns])


def test_calculate_costs(ymno3_rxns, rxn_set):
    cf = Softplus()
    assert np.allclose(
        np.sort(np.array(rxn_set.calculate_costs(cf))),
        np.sort(np.array([cf.evaluate(r) for r in ymno3_rxns])),
    )


def test_filter_duplicates(computed_rxn):
    computed_rxn2 = ComputedReaction(
        computed_rxn.entries, computed_rxn.coefficients * 2
    )

    assert computed_rxn2 != computed_rxn
    assert list(
        ReactionSet.from_rxns(
            [computed_rxn, computed_rxn2], filter_duplicates=True
        ).get_rxns()
    ) == [computed_rxn]
