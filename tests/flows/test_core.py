"""Tests for jobflow-based workflows"""

import pytest
from jobflow.managers.local import run_locally

from rxn_network.flows.core import NetworkFlowMaker, RetrosynthesisFlowMaker


@pytest.fixture
def network_flow_maker():
    """Create network flow maker"""
    return NetworkFlowMaker()


@pytest.fixture
def retrosynthesis_flow_maker():
    """Create retrosynthesis flow maker"""
    return RetrosynthesisFlowMaker()


@pytest.fixture
def network_flow_job(network_flow_maker, filtered_entries):
    """Create network flow job"""
    return network_flow_maker.make(
        ["Y2O3", "Mn2O3"], ["YMnO3", "Mn3O4"], entries=filtered_entries
    )


@pytest.fixture
def retrosynthesis_flow_job(retrosynthesis_flow_maker):
    """Create retrosynthesis flow job"""
    return retrosynthesis_flow_maker.make("YMnO3", entries=filtered_entries)


def test_network_flow_job(network_flow_job, job_store):
    output = run_locally(network_flow_job, store=job_store, ensure_success=True)

    doc = output[entry_job.uuid][1].output


def test_retrosynthesis_flow_job(retrosynthesis_flow_job, job_store):
    output = run_locally(retrosynthesis_flow_job, store=job_store, ensure_success=True)

    selectivities_doc = output[entry_job.uuid][-1].output
