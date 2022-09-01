import logging
from dataclasses import dataclass, field
from jobflow import Maker, Response, job
from rxn_network.jobs.models import EnumeratorTaskDocument, NetworkTaskDocument

logger = logging.getLogger(__name__)


# @dataclass
# class RetrosynthesisFlowMaker(Maker):
#     name: str = "enumerate reactions"
#     calculate_chempot_distance: bool = field(default=False)
#     entry_set_params: dict = field(default_factory=dict)

#     def make(self, enumerators, entries=None):
#         enumerator_task = EnumeratorTaskDocument.from_rxns_and_metadata(rxns, metadata)
#         return enumerator_task


# @dataclass
# class NetworkFlowMaker(Maker):
#     name: str = "build/analyze network"

#     def make(self, enumerators, entries=None):
#         network = build_network(enumerators, entries)
#         network_task = NetworkTaskDocument.from_network_and_metadata(network, metadata)
#         return network_task
