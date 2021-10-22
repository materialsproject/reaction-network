import pandas
from pymatgen.core.composition import Composition
import plotly.express as px

from rxn_network.costs.softplus import Softplus


def make_df_from_rxns(rxns, temp, target, cost_function):

    costs = [cost_function.evaluate(rxn) for rxn in rxns]
    target = Composition(target)
    if rxns[0].__class__.__name__ == "OpenComputedReaction":
        added_elems = [rxn.total_chemical_system != target.chemical_system for rxn in
                       rxns]
    else:
        added_elems = [rxn.chemical_system != target.chemical_system for rxn in rxns]
    df = pandas.DataFrame(
        {"rxn": rxns,
         "energy": [rxn.energy_per_atom for rxn in rxns],
         "dE": [rxn.energy_uncertainty_per_atom.s for rxn in rxns],
         "distance": [rxn.data["chempot_distance"] for rxn in rxns],
         "added_elems": added_elems,
         "cost": costs}
    ).sort_values("cost").reset_index(drop=True)

    return df


def plot_energy_distance_scatter(df):
    df = df.copy()
    df["rxn"] = df["rxn"].astype(str)

    fig = px.scatter(df,
                      x="energy",
                      y="distance",
                      hover_name="rxn",
                      labels={"energy":r"$\Delta G_{\mathrm{rxn}} ~ \mathrm{\left("
                                       r"\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$",
                              "distance":r"$\textrm{max(}\Delta \mu_{\mathrm{min}}) ~ \mathrm{\left(\dfrac{\mathsf{eV}}{\mathsf{atom}}\right)}$"},
                      hover_data={"energy":True,"distance":True},
                      error_x="dE",
                      template="simple_white",
                     color="added_elems",
                     width=800,
                     height=800)

    return fig


