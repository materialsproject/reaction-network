"""Core jobs for reaction-network creation and analysis."""

import logging
from dataclasses import dataclass, field
from jobflow import Maker, Response, job
from rxn_network.jobs.models import EnumeratorTaskDocument, NetworkTaskDocument

logger = logging.getLogger(__name__)


@dataclass
class EnumerationMaker(Maker):
    name: str = "enumerate reactions"
    calculate_chempot_distance: bool = field(default=False)
    entry_set_params: dict = field(default_factory=dict)

    @job
    def make(self, enumerators, entries=None):
        enumerator_task = EnumeratorTaskDocument.from_rxns_and_metadata(rxns, metadata)
        return enumerator_task


@dataclass
class RetrosynthesisMaker(Maker):
    name: str = "calculate synthesis recipes"

    @job
    def make(self, enumerators, entries=None):
        enumerator_task = EnumeratorTaskDocument.from_rxns_and_metadata(rxns, metadata)
        return enumerator_task


@dataclass
class NetworkMaker(Maker):
    name: str = "build/analyze network"

    @job
    def make(self, enumerators, entries=None):
        enumerator_task = NetworkTaskDocument.from_network_and_metadata(
            network, metadata
        )
        return network_task
