"""
Utility functions for plotting reaction data/analysis.
"""
from pandas import DataFrame
import plotly.express as px


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

    def get_label(name):
        label = ""
        if name == "energy":
            label = r"$\Delta G_{\mathrm{rxn}} ~ \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
        elif name == "chempot_distance":
            label = r"$\Sigma \Delta \mu_{\mathrm{min}} ~ \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"
        elif name == "c_score":
            label = "Competitiveness Score"
        return label

    df = df.copy()
    df["rxn"] = df["rxn"].astype(str)
    if "added_elems" in df:
        df["has_added_elems"] = df["added_elems"] != ""

    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_name="rxn",
        labels={i: get_label(i) for i in [x, y]},
        template="simple_white",
        color=color,
        color_discrete_map={True: "darkorange", False: "gray"},
        width=800,
        height=800,
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>"
        + "<br><b>energy</b>: %{x:.3f} (eV/atom)"
        + "<br><b>distance</b>: %{y:.3f} (eV/atom)<br>",
    )

    return fig


def plot_energy_distance_scatter(df: DataFrame) -> px.scatter:
    """
    Plot a Plotly scatter plot of chemical potential distance vs energy.

    Args:
        df: DataFrame with columns: rxn, energy, distance, added_elems

    Returns:
        Plotly scatter plot
    """

    df = df.copy()
    df["rxn"] = df["rxn"].astype(str)

    fig = px.scatter(
        df,
        x="energy",
        y="chempot_distance",
        hover_name="rxn",
        labels={
            "energy": r"$\Delta G_{\mathrm{rxn}} ~ \mathrm{\left("
            r"\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$",
            "chempot_distance": r"$\Sigma \Delta \mu_{\mathrm{min}}"
            r"~ \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$",
        },
        error_x="dE",
        template="simple_white",
        color="added_elems",
        color_discrete_sequence=["darkorange", "gray"],
        width=800,
        height=800,
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>"
        + "<br><b>energy</b>: %{x:.3f} (eV/atom)"
        + "<br><b>distance</b>: %{y:.3f} (eV/atom)<br>",
    )

    return fig
