""" Tests for GibbsEntrySet. """
from pathlib import Path

import pytest
from pytest import approx

from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PhaseDiagram

from rxn_network.entries.entry_set import GibbsEntrySet


TEST_FILES_PATH = Path(__file__).parent / "test_files"


@pytest.fixture
def mp_entries():
    mp_entries = loadfn(TEST_FILES_PATH / "Mn-O_entries.json")
    return mp_entries


@pytest.fixture
def gibbs_entries(mp_entries):
    entries = GibbsEntrySet.from_entries(mp_entries, temperature=1000)
    return entries


def test_from_pd(mp_entries):
    entries = GibbsEntrySet.from_pd(PhaseDiagram(mp_entries), temperature=1000)
    assert entries is not None
