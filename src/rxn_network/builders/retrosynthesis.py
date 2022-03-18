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

    def __init__(
        self, tasks: Store, tasks_fs: Store, recipes: Store, cf: CostFunction, **kwargs
    ):
        self.tasks = tasks
        self.tasks_fs = tasks_fs
        self.recipes = recipes
        self.cf = cf

        super().__init__(source=tasks, target=recipes, **kwargs)

    def get_items(self):
        """Get the items to process."""
        self.logger.info("Starting {} Builder".format(self.__class__.__name__))
        self.ensure_indexes()

        keys = self.target.newer_in(self.source, criteria=self.query, exhaustive=True)
        if self.retry_failed:
            if isinstance(self.query, (dict)):
                failed_query = {"$and": [self.query, {"state": "failed"}]}
            else:
                failed_query = {"state": "failed"}
            failed_keys = self.target.distinct(self.target.key, criteria=failed_query)
            keys = list(set(keys + failed_keys))

        self.logger.info("Processing {} items".format(len(keys)))

        if self.projection:
            projection = list(
                set(self.projection + [self.source.key, self.source.last_updated_field])
            )
        else:
            projection = None

        self.total = len(keys)
        for chunked_keys in grouper(keys, self.chunk_size):
            chunked_keys = list(chunked_keys)
            query = list(
                self.tasks.query(
                    criteria={self.tasks.key: {"$in": chunked_keys}},
                    properties=projection,
                )
            )
            query_fs = list(
                self.tasks_fs.query(
                    criteria={self.tasks_fs.key: {"$in": chunked_keys}},
                    properties=projection,
                )
            )
            for d1, d2 in zip(query, query_fs):
                d = d1
                d.update(d2)
                yield d1

        return self.tasks.find({"status": "COMPLETED"})

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
