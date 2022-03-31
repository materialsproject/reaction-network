""" Builder(s) for generating synthesis recipe documents."""
from datetime import datetime
from math import ceil
from typing import Any, Dict, Optional

from maggma.builders import Builder
from maggma.core import Store
from maggma.stores import GridFSStore
from maggma.utils import grouper
from monty.json import MontyDecoder, jsanitize
from pymatgen.core.composition import Composition

from rxn_network.core.cost_function import CostFunction
from rxn_network.utils.models import (
    ComputedSynthesisRecipe,
    ComputedSynthesisRecipesDoc,
)


class SynthesisRecipeBuilder(Builder):
    """
    Build a synthesis recipe document from the reaction results from EnumeratorWF.
    """

    def __init__(
        self,
        tasks: Store,
        recipes: Store,
        cf: CostFunction,
        tasks_fs: Optional[GridFSStore] = None,
        recipes_fs: Optional[GridFSStore] = None,
        query: Optional[Dict] = None,
        **kwargs,
    ):
        self.tasks = tasks
        self.tasks_fs = tasks_fs
        self.recipes = recipes
        self.recipes_fs = recipes_fs
        self.cf = cf
        self.query = query
        self.kwargs = kwargs

        sources = [tasks]
        targets = [recipes]

        if tasks_fs:
            sources.append(tasks_fs)
        if recipes_fs:
            targets.append(recipes_fs)

        super().__init__(sources=sources, targets=targets, **kwargs)

    def ensure_indexes(self):
        """
        Ensures indexes for the tasks, (tasks_fs), and recipes collections.
        """
        self.tasks.ensure_index(self.tasks.key)
        self.tasks.ensure_index(self.tasks.last_updated_field)
        self.recipes.ensure_index(self.recipes.key)
        self.recipes.ensure_index(self.recipes.last_updated_field)

        if self.tasks_fs:
            self.tasks_fs.ensure_index(self.tasks_fs.key)
        if self.recipes_fs:
            self.recipes_fs.ensure_index(self.recipes_fs.key)

    def prechunk(self, number_splits: int):
        """
        Prechunk method to perform chunking by the key field.
        """
        keys = self._find_to_process()

        N = ceil(len(keys) / number_splits)

        for split in grouper(keys, N):
            yield {"query": {self.tasks.key: {"$in": list(split)}}}

    def get_items(self):
        """Get the items to process."""
        to_process_task_ids = self._find_to_process()

        self.total = len(to_process_task_ids)
        self.logger.info(f"Processing {self.total} task docs for synthesis recipes")

        for task_id in to_process_task_ids:
            task = self.tasks.query_one({"task_id": task_id})
            if self.tasks_fs:
                rxns = self.tasks_fs.query_one(
                    {"task_id": task_id},
                )["rxns"]
                task["rxns"] = rxns
                if not rxns:
                    self.logger.warning(
                        f"Missing rxns from GridFSStore for task_id {task_id}"
                    )
            else:
                if not task.get("rxns"):
                    self.logger.warning(f"Missing rxns in task {task_id}")

            if task is not None:
                yield task
            else:
                pass

    def process_item(self, item):
        """Creates a synthesis recipe document from the task document."""
        item = MontyDecoder().process_decoded(item)

        task_id = item["task_id"]
        task_label = item["task_label"]
        rxns = item["rxns"]
        targets = item["targets"]
        elements = item["elements"]
        chemsys = item["chemsys"]
        added_elements = item["added_elements"]
        added_chemsys = item["added_chemsys"]
        enumerators = item["enumerators"]
        mu_func = None  # incorporate this?

        if len(targets) > 1:
            self.logger.warning(
                f"Enumerator has multiple targets for task_id {item['task_id']}"
            )
            self.logger.warning("Selecting first target...")

        target = item["targets"][0]
        target_comp = Composition(target)

        self.logger.debug(f"Creating synthesis recipes for {item['task_id']}")

        rxns = rxns.get_rxns()
        costs = [self.cf.evaluate(rxn) for rxn in rxns]
        recipes = [
            ComputedSynthesisRecipe.from_computed_rxn(
                rxn, cost=cost, target=target_comp, mu_func=mu_func
            )
            for rxn, cost in zip(rxns, costs)
        ]

        d: Dict[str, Any] = {}

        d["task_id"] = task_id
        d["task_label"] = task_label
        d["last_updated"] = datetime.utcnow()
        d["recipes"] = recipes
        d["target_composition"] = target_comp
        d["target_formula"] = target
        d["elements"] = elements
        d["chemsys"] = chemsys
        d["added_elements"] = added_elements
        d["added_chemsys"] = added_chemsys
        d["enumerators"] = enumerators
        d["cost_function"] = self.cf

        doc = ComputedSynthesisRecipesDoc(**d)

        return jsanitize(
            doc.dict(),
            strict=True,
            allow_bson=True,
        )

    def update_targets(self, items):
        """
        Inserts the new synthesis recipe docs into the Synthesis Recipes collection.
        Stores recipes in GridFS if a recipes GridFSStore is provided.
        """
        docs = list(filter(None, items))

        if len(docs) > 0:
            self.logger.info(f"Found {len(docs)} synthesis recipe docs to update")

            if self.recipes_fs:
                recipes = []
                for d in docs:
                    d["use_gridfs"] = True
                    recipe = {"task_id": d["task_id"], "recipes": d.pop("recipes")}
                    recipes.append(recipe)

                self.recipes_fs.update(
                    recipes, key="task_id", additional_metadata=["task_id"]
                )

            self.recipes.update(docs)
        else:
            self.logger.info("No items to update")

    def _find_to_process(self):
        self.logger.info("Synthesis Recipe builder started.")

        self.logger.info("Setting up indexes.")
        self.ensure_indexes()

        task_keys = set(self.tasks.distinct("task_id", criteria=self.query))
        updated_tasks = set(self.recipes.newer_in(self.tasks, exhaustive=True))
        return updated_tasks & task_keys
