"""Core jobs for reaction-network creation and analysis."""

import logging
import os
from typing import Optional
from dataclasses import dataclass, field
from maggma.stores import Store
from pymatgen.core.composition import Element
from jobflow import Maker, Response, job
from rxn_network.jobs.models import (
    EntrySetDocument,
    EnumeratorTaskDocument,
    NetworkTaskDocument,
)
from rxn_network.core.composition import Composition
from rxn_network.jobs.utils import (
    run_enumerators,
    build_network,
    run_solver,
)
from rxn_network.entries.utils import get_all_entries_in_chemsys, process_entries

logger = logging.getLogger(__name__)


@dataclass
class GetEntrySetMaker(Maker):
    """
    Maker to create job for acquiring and processing entries to be used in reaction
    enumeration or network building.
    """

    name: str = "get & process entries"
    db: Optional[Store] = None
    temperature: int = 300
    include_nist_data: bool = True
    include_barin_data: bool = False
    include_freed_data: bool = False
    e_above_hull: float = 0.0
    include_polymorphs: bool = False
    formulas_to_include: list = field(default_factory=list)

    @job(entries="entries")
    def make(self, chemsys):
        if self.db:
            entries = get_all_entries_in_chemsys(
                self.db,
                chemsys,
                inc_structure=True,
                compatible_only=True,
                property_data=None,
                use_premade_entries=False,
            )
        else:
            from mp_api import MPRester

            with MPRester() as mpr:
                entries = mpr.get_entries_in_chemsys(elements=chemsys)

        entries = process_entries(
            entries,
            temperature=self.temperature,
            include_nist_data=self.include_nist_data,
            include_barin_data=self.include_barin_data,
            include_freed_data=self.include_freed_data,
            e_above_hull=self.e_above_hull,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
        )

        doc = EntrySetDocument(
            entries=entries,
            e_above_hull=self.e_above_hull,
            include_polymorphs=self.include_polymorphs,
            formulas_to_include=self.formulas_to_include,
        )

        return entries


# @dataclass
# class ReactionEnumerationMaker(Maker):
#     name: str = "enumerate reactions"

#     @job(rxns="rxns")
#     def make(self, enumerators, entries):
#         rxns = run_enumerators(enumerators, entries)
#         metadata = self._get_metadata(enumerators, entries)

#         enumerator_task = EnumeratorTaskDocument.from_rxns_and_metadata(rxns, metadata)
#         return enumerator_task

#     def _get_metadata(self, enumerators, entries):
#         chemsys = "-".join(entries.chemsys)
#         targets = {
#             target for enumerator in enumerators for target in enumerator.targets
#         }

#         added_elements = None
#         added_chemsys = None

#         if targets:
#             added_elems = set(chemsys.split("-")) - {
#                 str(e) for target in targets for e in Composition(target).elements
#             }
#             added_chemsys = "-".join(sorted(list(added_elems)))
#             added_elements = [Element(e) for e in added_elems]

#         metadata = {
#             "elements": [Element(e) for e in chemsys.split("-")],
#             "chemsys": chemsys,
#             "enumerators": [e.as_dict() for e in enumerators],
#             "targets": list(sorted(targets)),
#             "added_elements": added_elements,
#             "added_chemsys": added_chemsys,
#         }
#         return metadata


# @dataclass
# class CalculateSelectivitiesMaker(Maker):
#     name: str = "calculate selectivities"
#     open_elem: Optional[Element] = None
#     chempot: Optional[float] = 0.0
#     calculate_primary_selectivity: bool = True
#     calculate_chempot_distance: bool = True
#     temp: float = 300.0
#     batch_size: int = 20

#     @job(rxns="rxns")
#     def make(self, rxn_sets, target_formula):
#         logger.info("Identifying target reactions...")

#         all_rxns = ReactionSet.from_rxns([r.get_rxns() for r in rxn_sets])

#         target_rxns = []
#         for rxn in all_rxns:
#             product_formulas = [p.reduced_formula for p in rxn.products]
#             if target_formula in product_formulas:
#                 target_rxns.append(rxn)

#         logger.info(
#             f"Identified {len(target_rxns)} target reactions out of"
#             f" {len(all_rxns)} total reactions."
#         )

#         all_rxns = ray.put(all_rxns)

#         logger.info("Calculating selectivites...")

#         processed_chunks = []
#         for rxns_chunk in grouper(target_rxns, batch_size, fillvalue=None):
#             processed_chunks.append(
#                 get_decorated_rxns_by_chunk.remote(
#                     rxns_chunk, all_rxns, open_formula, temp
#                 )
#             )

#         decorated_rxns = []
#         for r in tqdm(
#             to_iterator(processed_chunks),
#             total=len(processed_chunks),
#             desc="Selectivity",
#         ):
#             decorated_rxns.extend(r)

#         del processed_chunks

#         logger.info("Saving decorated reactions.")

#         if self.calculate_chempot_distance:

#         cpd_kwargs = self.get("cpd_kwargs", {})
#         results = self._get_decorated_rxns(rxns, entries, cpd_kwargs)

#         metadata["cpd_kwargs"] = cpd_kwargs
#         dumpfn(metadata, "metadata.json.gz")  # will overwrite existing metadata.json.gz

#         results = ReactionSet.from_rxns(decorated_rxns)
#         dumpfn(results, "rxns.json.gz")  # may overwrite existing rxns.json.gz


# @ray.remote
# def get_decorated_rxns_by_chunk(rxn_chunk, all_rxns, open_formula, temp):
#     decorated_rxns = []

#     for rxn in rxn_chunk:
#         if not rxn:
#             continue

#         precursors = [r.reduced_formula for r in rxn.reactants]
#         competing_rxns = list(all_rxns.get_rxns_by_reactants(precursors))

#         if open_formula:
#             open_formula = Composition(open_formula).reduced_formula
#             competing_rxns.extend(
#                 all_rxns.get_rxns_by_reactants(precursors + [open_formula])
#             )

#         if len(precursors) >= 3:
#             precursors = list(set(precursors) - {open_formula})

#         decorated_rxns.append(get_decorated_rxn(rxn, competing_rxns, precursors, temp))

#     return decorated_rxns


# # @dataclass
# # class NetworkMaker(Maker):
# #     name: str = "build/analyze network"

# #     @job
# #     def make(self, enumerators, entries=None):
# #         network = build_network(enumerators, entries)
# #         enumerator_task = NetworkTaskDocument.from_network_and_metadata(
# #             network, metadata
# #         )
# #         return network_task
