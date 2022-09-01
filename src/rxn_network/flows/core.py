import logging
from dataclasses import dataclass, field
from typing import Optional

from jobflow import Maker, Response, job, Flow
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.jobs.core import (
    CalculateSelectivitiesMaker,
    ReactionEnumerationMaker,
    GetEntrySetMaker,
    NetworkMaker,
)
from rxn_network.jobs.models import EnumeratorTaskDocument, NetworkTaskDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrosynthesisFlowMaker(Maker):
    name: str = "enumerate reactions"
    calculate_chempot_distance: bool = field(default=False)
    get_entry_set_maker: GetEntrySetMaker = field(default_factory=GetEntrySetMaker)
    enumeration_maker: ReactionEnumerationMaker = field(
        default_factory=ReactionEnumerationMaker
    )
    calculate_selectivities_maker: Optional[CalculateSelectivitiesMaker] = None
    open_elem: Optional[Element] = None
    chempot: Optional[float] = None

    def make(self, target_formula, added_elems):
        jobs = []

        chemsys = "-".join(
            {str(e) for e in Composition(target_formula).elements}
            | {str(e) for e in added_elems}
        )

        get_entry_set_job = self.get_entry_set_maker.make(chemsys)
        jobs.append(get_entry_set_job)

        enumerators = [BasicEnumerator(targets=[target_formula])]

        enumeration_job = self.enumeration_maker.make(
            enumerators=enumerators, entries=get_entry_set_job.output["entries"]
        )
        jobs.append(enumeration_job)

        calculate_selectivities_job = self.calculate_selectivities_maker.make(
            rxn_sets=[enumeration_job.output["rxns"]],
            entries=get_entry_set_job.output["entries"],
            target_formula=target_formula,
        )
        jobs.append(calculate_selectivities_job)

        return Flow(jobs, name="Retrosynthesis Flow")


# @dataclass
# class NetworkFlowMaker(Maker):
#     name: str = "build/analyze network"

#     def make(self, enumerators, entries=None):
#         network = build_network(enumerators, entries)
#         network_task = NetworkTaskDocument.from_network_and_metadata(network, metadata)
#         return network_task
