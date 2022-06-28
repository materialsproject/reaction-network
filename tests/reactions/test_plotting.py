""" Tests for reaction plotting functions."""
from pathlib import Path
import pytest

from rxn_network.costs.softplus import Softplus
from rxn_network.reactions.plotting import plot_reaction_scatter
from rxn_network.reactions.reaction_set import ReactionSet


@pytest.fixture(scope="module")
def df(ymno_rxns):
    return ReactionSet.from_rxns(ymno_rxns).to_dataframe(Softplus())


def test_plot_reaction_scatter(df):
    plot = plot_reaction_scatter(df, x="energy", y="dE", color="dE")
