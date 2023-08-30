"""
Utility functions for plotting reaction data & performing analysis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
from pymatgen.analysis.chempot_diagram import plotly_layouts

from rxn_network.costs.pareto import get_pareto_front

if TYPE_CHECKING:
    from pandas import DataFrame


def plot_reaction_scatter(
    df: DataFrame,
    x: str = "secondary_competition",
    y: str = "energy",
    z: str | None = None,
    color: str | None = None,
    plot_pareto: bool = True,
) -> px.scatter:
    """
    Plot a Plotly scatter plot (2D or 3D) of reactions on various thermodynamic metric
    axes. This also constructs the Pareto front on the provided dimensions.

    Args:
        df: DataFrame with columns: rxn, energy, (primary_competition),
            (secondary_competition), (chempot_distance), (added_elems), (dE)

    Returns:
        Plotly scatter plot
    """

    def get_label_and_units(name):
        label, units = "", ""

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
        elif name == "primary_competition":
            label = "Primary Competition"
            units = "eV/atom"
        elif name == "secondary_competition":
            label = "Secondary Competition"
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
        pareto_df = get_pareto_front(df, metrics=cols)
        df = df.loc[~df.index.isin(pareto_df.index)]

        arr = pareto_df[list(cols)].to_numpy()
        if z is None:
            scatter = go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                hovertext=pareto_df["rxn"],
                marker={"size": 10, "color": "seagreen", "symbol": "diamond"},
                mode="markers",
                name="Pareto front",
            )
        else:
            scatter = go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                hovertext=pareto_df["rxn"],
                marker={"size": 10, "color": "seagreen", "symbol": "diamond"},
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

        layout_3d["scene_camera"] = {
            "eye": {"x": -5, "y": -5, "z": 5},  # zoomed out
            "projection": {"type": "orthographic"},
            "center": {"x": -0.2, "y": -0.2, "z": -0.1},
        }

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


def pretty_df_layout(df: DataFrame):
    """Improve visibility for a pandas DataFrame with wide column names"""
    return df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("max-width", "70px"),
                    ("text-overflow", "ellipsis"),
                    ("overflow", "hidden"),
                ],
            }
        ]
    )  # improve rendering in Jupyter


def filter_df_by_precursors(df: DataFrame, precursors: list[str]):
    """Filter a reaction DataFrame by available precursors"""
    df = df.copy()
    df["precursors"] = [
        list(sorted([r.reduced_formula for r in rxn.reactants])) for rxn in df["rxn"]
    ]
    selected = df[df["precursors"].apply(lambda x: all(p in precursors for p in x))]
    return selected.drop(columns=["precursors"])
