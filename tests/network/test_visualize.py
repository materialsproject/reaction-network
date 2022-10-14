"""Test network visualization"""

import pytest

from rxn_network.network.visualize import plot_network


@pytest.mark.skip(reason="Gtk issues on CI")
def test_plot_network(ymno_rn):
    """Test plot_network"""
    plot_network(ymno_rn.graph)
