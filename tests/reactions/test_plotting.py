""" Tests for reaction plotting functions."""
import pytest

from rxn_network.costs.softplus import Softplus
from rxn_network.reactions.plotting import plot_reaction_scatter
from rxn_network.reactions.reaction_set import ReactionSet


@pytest.fixture(scope="module")
def df(ymno3_rxns):
    rxn_set = ReactionSet.from_rxns(ymno3_rxns)
    return rxn_set.to_dataframe(Softplus(), calculate_uncertainties=True)


def test_plot_reaction_scatter(df):
    plot = plot_reaction_scatter(df, x="energy", y="dE", color="dE")
