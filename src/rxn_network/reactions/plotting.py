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
                r"$\textrm{Reaction driving force} ~"
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
                hovertext=df["rxn"],
                marker=dict(size=10, color="green"),
                mode="markers",
            )
        else:
            scatter = go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                hovertext=df["rxn"],
                marker=dict(size=10, color="green"),
                mode="markers",
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
            color_discrete_map={True: "darkorange", False: "gray"},
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
            color_discrete_map={True: "darkorange", False: "gray"},
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
            hovertemplate + "<br><b>" + f"{z}" + "</b>: %{y:.3f}" + f" {z_units}<br>"
        )

    fig.update_traces(hovertemplate=hovertemplate)

    return fig


def get_pareto_front(
    df: DataFrame,
    cols=("energy", "primary_selectivity", "secondary_selectivity"),
    minimize=True,
) -> DataFrame:
    """

    Returns a new reaction DataFrame containing only reactions on the Pareto front for
    the specified columns (i.e., reaction parameters)

    This function has been adapted from user @hilberts_drinking_problem on
    StackOverflow. Thanks for the great answer!

    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    """
    df_original = df.copy()
    df = df_original[list(cols)]
    pts = df.reset_index().to_numpy()

    if minimize:
        pts[:, 1:] = pts[:, 1:] * -1

    pts = pts[pts[:, 1:].sum(axis=1).argsort()[::-1]]
    undominated = np.ones(pts.shape[0], dtype=bool)

    for i in range(pts.shape[0]):
        n = pts.shape[0]
        if i >= n:
            break
        undominated[i + 1 : n] = (pts[i + 1 :, 1:] >= pts[i, 1:]).any(1)
        pts = pts[undominated[:n]]

    return df_original.loc[pts[:, 0]]
