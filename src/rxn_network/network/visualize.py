"""
Functions for visualizing/plotting reaction networks.
"""

import graph_tool.all as gt
import matplotlib.cm
import numpy as np


def plot_network(
    graph,
    vertex_cmap="jet",
    edge_cmap="PuBuGn_r",
    output=None,
    cost_pos_scale_factor=10,
):
    g = graph.copy()

    costs = np.array(g.ep["cost"].get_array().tolist())

    edge_weights = gt.prop_to_size(
        g.new_edge_property("float", cost_pos_scale_factor * costs), mi=0.1, ma=35
    )

    deg = gt.prop_to_size(g.degree_property_map("total"), mi=0.5, ma=22)
    layout = gt.sfdp_layout(g, vweight=deg, eweight=edge_weights)

    chemsys_names = [g.vp["entry"][v].chemsys for v in g.vertices()]

    edge_width = [1.4 if g.ep["cost"][e] != 0 else 0.1 for e in g.edges()]
    edge_width = g.new_edge_property("float", edge_width)

    color_func_v = _get_cmap_string(vertex_cmap, domain=sorted(chemsys_names))
    vertex_colors = [color_func_v(chemsys) for chemsys in chemsys_names]
    vertex_colors = g.new_vertex_property("vector<float>", vertex_colors)

    avg_cost = np.mean(costs)
    vmin = avg_cost - 0.8 * avg_cost
    vmax = avg_cost + 0.8 * avg_cost
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    edge_cmap = matplotlib.cm.get_cmap(edge_cmap)
    edge_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=edge_cmap)

    edge_colors = [edge_cmap.to_rgba(cost, alpha=0.5) for cost in costs]
    edge_colors = g.new_edge_property("vector<float>", edge_colors)

    control = g.new_edge_property("vector<double>")
    for e in g.edges():
        d = np.sqrt(sum((layout[e.source()].a - layout[e.target()].a) ** 2)) / 20
        control[e] = [0.0, 0.0, 0.3, d, 0.7, d, 1.0, 0.0]

    return gt.graph_draw(
        g,
        pos=layout,
        vertex_fill_color=vertex_colors,
        edge_color=edge_colors,
        edge_pen_width=edge_width,
        vertex_size=deg,
        edge_control_points=control,
        output=output,
    )


def plot_network_on_graphistry(graph):
    try:
        import graphistry
        import networkx as nx
        import pyintergraph
    except ImportError:
        print(
            "Must install optional dependencies: pygraphistry, networkx, "
            "and pyintergraph!"
        )
        return

    nx_graph = pyintergraph.gt2nx(graph)
    mapping = {}

    for node in nx_graph.nodes(data=True):
        mapping[node[0]] = str(node[1]["entry"])
        nx_graph.nodes()[node[0]]["entry"] = str(nx_graph.nodes()[node[0]]["entry"])

    nx.relabel_nodes(nx_graph, mapping, copy=False)

    for edge in nx_graph.edges:
        nx_graph.edges[edge]["cost"] = float(nx_graph.edges[edge]["cost"])
        if nx_graph.edges[edge]["rxn"]:
            nx_graph.edges[edge]["rxn"] = str(nx_graph.edges[edge]["rxn"])
        else:
            nx_graph.edges[edge]["rxn"] = "None"

    return graphistry.bind(source="src", destination="dst", node="nodeid").plot(
        nx_graph, render=True
    )


def _get_cmap_string(palette, domain):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out
