"""
Graph-related functions specific to the graph-tool library. Used in the network module.
"""

from queue import Empty, PriorityQueue
from typing import Dict

from graph_tool import Graph, GraphView, Vertex
from graph_tool.topology import shortest_path

DEFAULT_VERTEX_PROPS = {"entry": "object", "type": "int"}
DEFAULT_EDGE_PROPS = {"rxn": "object", "cost": "double", "type": "string"}


def initialize_graph(
    vertex_props: Dict[str, str] = None, edge_props: Dict[str, str] = None
) -> Graph:
    """

    Args:
        vertex_props:
        edge_props:

    Returns:

    """
    g = Graph()

    if not vertex_props:
        vertex_props = dict()
    if not edge_props:
        edge_props = dict()

    vertex_props.update(DEFAULT_VERTEX_PROPS)
    edge_props.update(DEFAULT_EDGE_PROPS)

    for name, obj_type in vertex_props.items():
        g.vp[name] = g.new_vertex_property(obj_type)
    for name, obj_type in edge_props.items():
        g.ep[name] = g.new_edge_property(obj_type)

    return g


def update_vertex_props(g, v, prop_dict):
    """
    Helper method for updating several vertex properties at once in a graph-tool
    graph.

    Args:
        g (gt.Graph): a graph-tool Graph object.
        v (gt.Vertex or int): a graph-tool Vertex object (or its index) for a vertex
            in the provided graph.
        prop_dict (dict): a dictionary of the form {"prop": val}, where prop is the
            name of a VertexPropertyMap of the graph and val is the new updated
            value for that vertex's property.

    Returns:
        None
    """
    for prop, val in prop_dict.items():
        g.vp[prop][v] = val


def yens_ksp(
    g: Graph,
    num_k: int,
    precursors_v: Vertex,
    target_v: Vertex,
    edge_prop: str = "bool",
    cost_prop: str = "cost",
):
    """
    Yen's Algorithm for k-shortest paths, adopted for graph-tool. Utilizes GraphView
    objects to speed up filtering. Inspired by igraph implementation by
    Antonin Lenfant.

    Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths
    in a Network", Management Science, Vol. 17, No. 11, Theory Series (Jul.,
    1971), pp. 712-716.

    Args:
        g: the graph-tool graph object.
        num_k: number of k shortest paths that should be found.
        precursors_v: graph-tool vertex object containing precursors.
        target_v: graph-tool vertex object containing target.
        edge_prop: name of edge property map which allows for filtering edges.
            Defaults to the word "bool".
        cost_prop: name of edge property map that stores edge weights/costs.
            Defaults to the word "weight".
    Returns:
        List of lists of graph vertices corresponding to each shortest path
            (sorted in increasing order by cost).
    """

    def path_cost(vertices):
        """Calculates path cost given a list of vertices."""
        cost = 0
        for j in range(len(vertices) - 1):
            cost += g.ep[cost_prop][g.edge(vertices[j], vertices[j + 1])]
        return cost

    path = shortest_path(g, precursors_v, target_v, weights=g.ep[cost_prop])[0]

    if not path:
        return []
    a = [path]
    a_costs = [path_cost(path)]

    b = PriorityQueue()  # type: ignore

    g.ep["bool"] = g.new_edge_property("bool", val=True)

    for k in range(1, num_k):
        try:
            prev_path = a[k - 1]
        except IndexError:
            print(f"Identified only k={k-1} paths before exiting. \n")
            break

        for i in range(len(prev_path) - 1):
            spur_v = prev_path[i]
            root_path = prev_path[:i]

            filtered_edges = []

            for path in a:
                if len(path) - 1 > i and root_path == path[:i]:
                    e = g.edge(path[i], path[i + 1])
                    if not e:
                        continue
                    g.ep[edge_prop][e] = False
                    filtered_edges.append(e)

            gv = GraphView(g, efilt=g.ep[edge_prop])
            spur_path = shortest_path(gv, spur_v, target_v, weights=g.ep[cost_prop])[0]

            for e in filtered_edges:
                g.ep[edge_prop][e] = True

            if spur_path:
                total_path = root_path + spur_path
                total_path_cost = path_cost(total_path)
                b.put((total_path_cost, total_path))

        while True:
            try:
                cost_, path_ = b.get(block=False)
            except Empty:
                break
            if path_ not in a:
                a.append(path_)
                a_costs.append(cost_)
                break

    return a


def update_vertex_properties(g, v, prop_dict):
    """
    Helper method for updating several vertex properties at once in a graph-tool
    graph.

    Args:
        g (gt.Graph): a graph-tool Graph object.
        v (gt.Vertex or int): a graph-tool Vertex object (or its index) for a vertex
            in the provided graph.
        prop_dict (dict): a dictionary of the form {"prop": val}, where prop is the
            name of a VertexPropertyMap of the graph and val is the new updated
            value for that vertex's property.

    Returns:
        None
    """
    for prop, val in prop_dict.items():
        g.vp[prop][v] = val
    return None
