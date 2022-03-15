from typing import Dict, List, Optional

from maggma.builders.map_builder import MapBuilder
from maggma.core import Store
from pymatgen.core.composition import Composition

from rxn_network.core.cost_function import CostFunction
from rxn_network.reactions.reaction_set import ReactionSet


class SynthesisRecipeBuilder(MapBuilder):
    """
    Build a synthesis recipe document from the reaction results from EnumeratorWF.
    """

    def __init__(self, tasks: Store, recipes: Store, cf: CostFunction, **kwargs):
        self.tasks = tasks
        self.recipes = recipes
        self.cf = cf

        super().__init__(source=tasks, target=recipes, **kwargs)

    def unary_function(self, item):
        rxns_dict = item["rxns"]
        target = Composition(item["targets"][0])
        added_elems = item["added_elems"]
        enumerators = item["enumerators"]

        rxn_set = ReactionSet.from_dict(rxns_dict)
        df = rxn_set.to_dataframe(cost_function=self.cf, target=target)

        d = {}
        d["recipes"] = df.to_dict(orient="records")
        d["target"] = target
        d["target_formula"] = target.reduced_formula
        d["added_elems"] = added_elems
        d["enumerators"] = enumerators
        d["cost_function"] = self.cf.as_dict()

        return d
