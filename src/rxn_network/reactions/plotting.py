import pandas
import plotly.express as px
from pymatgen.core.composition import Composition

from rxn_network.costs.softplus import Softplus


def plot_energy_distance_scatter(df) -> px.scatter:
    df = df.copy()
    df["rxn"] = df["rxn"].astype(str)

    fig = px.scatter(
        df,
        x="energy",
        y="distance",
        hover_name="rxn",
        labels={
            "energy": r"$\Delta G_{\mathrm{rxn}} ~ \mathrm{\left("
            r"\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$",
            "distance": r"$\textrm{max(}\Delta \mu_{\mathrm{min}})"
            r"~ \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$",
        },
        hover_data={"energy": True, "distance": True},
        error_x="dE",
        template="simple_white",
        color="added_elems",
        width=800,
        height=800,
    )

    return fig
