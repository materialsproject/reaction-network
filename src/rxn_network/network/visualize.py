"""
Functions for visualizing/plotting reaction networks.
"""
import warnings
from typing import Optional

import matplotlib.cm
import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw


def plot_network(graph: rx.PyGraph, vertex_cmap_name: str = "jet", **kwargs):
    """
    Plots a reaction network using graph-tool visualization tools (i.e., graph_draw())

    Args:
        graph: a graph-tool Graph object
        vertex_cmap_name: the name of . Defaults to "jet".
        edge_cmap_name: Defaults to "PuBuGn_r".
        output: Optional output filename
        cost_pos_scale_factor

    """
    g = graph.copy()

    node_names = [e.chemsys for e in g.nodes()]
    color_func_v = _get_cmap_string(vertex_cmap_name, domain=sorted(node_names))
    vertex_colors = [color_func_v(chemsys) for chemsys in node_names]

    return mpl_draw(
        g,
        node_size=2,
        width=0.2,
        arrow_size=3,
        node_color=vertex_colors,
        alpha=0.8,
        **kwargs
    )


def _get_cmap_string(palette, domain):
    """
    Utility function for getting a matplotlib colormap string for a given palette and
    domain.
    """
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out
