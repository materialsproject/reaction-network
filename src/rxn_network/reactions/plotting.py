"""
Utility functions for plotting reaction data/analysis.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame
from pymatgen.analysis.chempot_diagram import plotly_layouts


def plot_reaction_scatter(
    df: DataFrame,
    x="energy",
    y="secondary_selectivity",
    z=None,
    color="has_added_elems",
    plot_pareto=True,
) -> px.scatter:
    """
    Plot a Plotly scatter plot of chemical potential distance vs energy.

    Args:
        df: DataFrame with columns: rxn, energy, distance, added_elems

    Returns:
        Plotly scatter plot
    """

    def get_label_and_units(name):
        label = ""
        units = ""
        if name == "energy":
            label = (
                r"$\mathsf{Reaction~driving~force} ~"
                r"\mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
            )
            units = "eV/atom"
            if z is not None:
                label = "Reaction Driving Force"
        elif name == "chempot_distance":
            label = (
                r"$\Sigma \Delta \mu_{\mathrm{min}} ~"
                r" \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
            )
            if z is not None:
                label = "Total chemical potential distance"
            units = "eV/atom"
        elif name == "primary_selectivity":
            label = "Primary Selectivity"
            units = "a.u."
        elif name == "secondary_selectivity":
            label = "Secondary Selectivity"
            units = "eV/atom"
        elif name == "dE":
            label = "Uncertainty"
            units = "eV/atom"
        return label, units

    df = df.copy()
    df["rxn"] = df["rxn"].astype(str)
    if "added_elems" in df:
        df["has_added_elems"] = df["added_elems"] != ""

    x_label, x_units = get_label_and_units(x)
    y_label, y_units = get_label_and_units(y)
    z_label, z_units = None, None

    cols: tuple = (x, y)

    if z is not None:
        z_label, z_units = get_label_and_units(z)
        cols = (x, y, z)

    if plot_pareto:
        pareto_df = get_pareto_front(df, cols=cols)
        df = df.loc[~df.index.isin(pareto_df.index)]

        arr = pareto_df[list(cols)].to_numpy()
        if z is None:
            scatter = go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                hovertext=pareto_df["rxn"],
                marker=dict(size=10, color="seagreen", symbol="diamond"),
                mode="markers",
                name="Pareto front",
            )
        else:
            scatter = go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                hovertext=pareto_df["rxn"],
                marker=dict(size=10, color="seagreen", symbol="diamond"),
                mode="markers",
                name="Pareto front",
            )

    if z is None:
        layout_2d = plotly_layouts["default_layout_2d"]
        fig = px.scatter(
            df,
            x=x,
            y=y,
            hover_name="rxn",
            labels={x: x_label, y: y_label},
            color=color,
            color_discrete_map={True: "darkorange", False: "lightgray"},
        )
        fig.update_layout(layout_2d)
    else:
        layout_3d = plotly_layouts["default_layout_3d"]
        axis_layout = plotly_layouts["default_3d_axis_layout"].copy()
        axis_layout["titlefont"]["size"] = 14
        for t in ["xaxis", "yaxis", "zaxis"]:
            layout_3d["scene"][t] = axis_layout

        layout_3d["scene_camera"] = dict(
            eye=dict(x=-5, y=-5, z=5),  # zoomed out
            projection=dict(type="orthographic"),
            center=dict(x=-0.2, y=-0.2, z=-0.1),
        )

        fig = px.scatter_3d(
            df,
            x=x,
            y=y,
            z=z,
            hover_name="rxn",
            labels={x: x_label, y: y_label, z: z_label},
            template="simple_white",
            color=color,
            color_discrete_map={True: "darkorange", False: "lightgray"},
        )

        fig.update_layout(layout_3d)

    if plot_pareto:
        fig.add_trace(scatter)

    hovertemplate = (
        "<b>%{hovertext}</b><br>"
        + "<br><b>"
        + f"{x}"
        + "</b>: %{x:.3f}"
        + f" {x_units}"
        + "<br><b>"
        + f"{y}"
        + "</b>: %{y:.3f}"
        + f" {y_units}"
    )

    if z is not None:
        hovertemplate = (
            hovertemplate + "<br><b>" + f"{z}" + "</b>: %{z:.3f}" + f" {z_units}<br>"
        )

    fig.update_traces(hovertemplate=hovertemplate)

    return fig


def get_pareto_front(
    df: DataFrame,
    cols=("energy", "primary_selectivity", "secondary_selectivity"),
    maximize=False,
):

    df_original = df.copy()
    df = df_original[list(cols)]
    pts = df.to_numpy()

    if maximize:
        pts[:, 1:] = pts[:, 1:] * -1

    return df_original[is_pareto_efficient(pts, return_mask=True)]


def is_pareto_efficient(costs, return_mask=True):
    """
    Directly borrowed from @Peter's numpy-based solution on stackoverflow. Please
    give him an upvote here: https://stackoverflow.com/a/40239615.
    Thank you @Peter!

    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

    return is_efficient


def pretty_df_layout(df):
    """Improve visibility for a pandas DataFrame with wide column names"""
    return df.style.set_table_styles(
        [
            dict(
                selector="th",
                props=[
                    ("max-width", "70px"),
                    ("text-overflow", "ellipsis"),
                    ("overflow", "hidden"),
                ],
            )
        ]
    )  # improve rendering in Jupyter


def filter_df_by_precursors(df, precursors):
    """Filter a reaction DataFrame by available precursors"""
    df = df.copy()
    df["precursors"] = [
        list(sorted([r.reduced_formula for r in rxn.reactants])) for rxn in df["rxn"]
    ]
    selected = df[df["precursors"].apply(lambda x: all(p in precursors for p in x))]
    return selected.drop(columns=["precursors"])
