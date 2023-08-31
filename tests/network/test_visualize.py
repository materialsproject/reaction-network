"""Test network visualization"""

from rxn_network.network.visualize import plot_network


def test_plot_network(ymno_rn):
    """Test plot_network"""
    plot_network(ymno_rn.graph)
