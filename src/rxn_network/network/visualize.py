import graphistry
import pyintergraph
import networkx as nx
import graph_tool.all as gt


def plot_network(g):
    return gt.graph_draw(g)

def plot_network_on_graphistry(g):
    nx_graph = pyintergraph.gt2nx(g)
    mapping = {}

    for node in nx_graph.nodes(data=True):
        mapping[node[0]] = str(node[1]['entry'])
        nx_graph.nodes()[node[0]]['entry'] = str(nx_graph.nodes()[node[0]]['entry'])

    nx.relabel_nodes(nx_graph, mapping, copy=False)

    for edge in nx_graph.edges:
        nx_graph.edges[edge]["cost"] = float(nx_graph.edges[edge]["cost"])
        if nx_graph.edges[edge]["rxn"]:
            nx_graph.edges[edge]["rxn"] = str(nx_graph.edges[edge]["rxn"])
        else:
            nx_graph.edges[edge]["rxn"] = "None"

    return graphistry.bind(source='src', destination='dst', node='nodeid').plot(
        nx_graph, render=True)
