""" Tests for ReactionSet."""
import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core.composition import Element

from rxn_network.core import Composition
from rxn_network.costs.functions import Softplus
from rxn_network.reactions.computed import ComputedReaction
from rxn_network.reactions.open import OpenComputedReaction
from rxn_network.reactions.reaction_set import ReactionSet


@pytest.fixture(scope="module")
def rxn_set(ymno3_rxns):
    return ReactionSet.from_rxns(ymno3_rxns)


@pytest.fixture(scope="module")
def open_rxn_set(ymno3_rxns):
    return ReactionSet.from_rxns(ymno3_rxns, open_elem="O", chempot=0)


@pytest.fixture(scope="module")
def gibbs_rxn_set(ymno3_gibbs_rxns):
    return ReactionSet.from_rxns(ymno3_gibbs_rxns)


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


def test_set_temperature(gibbs_rxn_set: ReactionSet):
    # ymno3_rxns was calculated at an unknown temperature, so we choose 1234
    temp = 1234

    new_rxn_set = gibbs_rxn_set.set_new_temperature(temp)

    new_rxns = list(new_rxn_set.get_rxns())
    old_rxns = list(gibbs_rxn_set.get_rxns())

    assert len(new_rxns) == len(old_rxns)

    old_test_rxn = old_rxns[0]
    found = [
        rxn
        for rxn in new_rxns
        if rxn.reactants == old_test_rxn.reactants
        and np.all(rxn.coefficients == old_test_rxn.coefficients)
    ]

    assert len(found) == 1

    new_test_rxn = found[0]

    assert not old_test_rxn.energy_per_atom == new_test_rxn.energy_per_atom


def test_set_open_el(rxn_set: ReactionSet):
    open_el = Element("O")
    chempot = -0.5

    new_rxn_set = rxn_set.set_chempot(open_el, chempot)

    new_rxns = list(new_rxn_set.get_rxns())
    old_rxns = list(rxn_set.get_rxns())

    assert len(new_rxns) == len(old_rxns)

    assert all([type(r) == OpenComputedReaction for r in new_rxns])
    assert all([r.chempots == {open_el: chempot} for r in new_rxns])

    old_test_rxn = old_rxns[0]
    found = [
        rxn
        for rxn in new_rxns
        if rxn.reactants == old_test_rxn.reactants
        and np.all(rxn.coefficients == old_test_rxn.coefficients)
    ]

    assert len(found) == 1

    new_test_rxn = found[0]

    assert not old_test_rxn.energy_per_atom == new_test_rxn.energy_per_atom


def test_unset_open_el(open_rxn_set: ReactionSet):
    open_el = None
    chempot = None

    new_rxn_set = open_rxn_set.set_chempot(open_el, chempot)

    new_rxns = list(new_rxn_set.get_rxns())
    old_rxns = list(open_rxn_set.get_rxns())

    assert len(new_rxns) == len(old_rxns)

    assert all([type(r) == ComputedReaction for r in new_rxns])
