from queue import PriorityQueue
from graph_tool import Graph, Vertex, GraphView
from graph_tool.topology import shortest_path


def yens_ksp(
        g: Graph,
        num_k: int,
        precursors_v: Vertex,
        target_v: Vertex,
        edge_prop: str="bool",
        weight_prop: str="weight"
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
        weight_prop: name of edge property map that stores edge weights/costs.
            Defaults to the word "weight".
    Returns:
        List of lists of graph vertices corresponding to each shortest path
            (sorted in increasing order by cost).
    """

    def path_cost(vertices):
        """Calculates path cost given a list of vertices."""
        cost = 0
        for j in range(len(vertices) - 1):
            cost += g.ep[weight_prop][g.edge(vertices[j], vertices[j + 1])]
        return cost

    path = shortest_path(g, precursors_v, target_v, weights=g.ep[weight_prop])[0]

    if not path:
        return []
    a = [path]
    a_costs = [path_cost(path)]

    b = PriorityQueue()  # automatically sorts by path cost (priority)

    for k in range(1, num_k):
        try:
            prev_path = a[k - 1]
        except IndexError:
            print(f"Identified only k={k} paths before exiting. \n")
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
            spur_path = shortest_path(
                gv, spur_v, target_v, weights=g.ep[weight_prop]
            )[0]

            for e in filtered_edges:
                g.ep[edge_prop][e] = True

            if spur_path:
                total_path = root_path + spur_path
                total_path_cost = path_cost(total_path)
                b.put((total_path_cost, total_path))

        while True:
            try:
                cost_, path_ = b.get(block=False)
            except queue.Empty:
                break
            if path_ not in a:
                a.append(path_)
                a_costs.append(cost_)
                break

    return a