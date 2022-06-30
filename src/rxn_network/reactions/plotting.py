"""
Utility functions for plotting reaction data/analysis.
"""
import plotly.express as px
from pandas import DataFrame


def plot_reaction_scatter(
    df: DataFrame, x="energy", y="chempot_distance", color="has_added_elems"
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
                r"$\Delta G_{\mathrm{rxn}} ~"
                r" \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
            )
            units = "eV/atom"
        elif name == "chempot_distance":
            label = (
                r"$\Sigma \Delta \mu_{\mathrm{min}} ~"
                r" \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
            )
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

    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_name="rxn",
        labels={x: x_label, y: y_label},
        template="simple_white",
        color=color,
        color_discrete_map={True: "darkorange", False: "gray"},
        width=800,
        height=800,
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>"
        + "<br><b>"
        + f"{x}"
        + "</b>: %{x:.3f}"
        + f" {x_units}"
        + "<br><b>"
        + f"{y}"
        + "</b>: %{y:.3f}"
        + f" {y_units}<br>",
    )

    return fig
