"""Tests for FreedReferenceEntry"""

import pytest

from rxn_network.core import Composition
from rxn_network.entries.freed import FREEDReferenceEntry


@pytest.fixture
def freed_entry():
    return FREEDReferenceEntry(Composition("H2O"), 300)


def test_repr(freed_entry):
    assert (
        repr(freed_entry) == "FREEDReferenceEntry | H2O\nGibbs Energy (300 K) = -2.4552"
    )


def test_eq(freed_entry, mp_entries):
    assert freed_entry == FREEDReferenceEntry(Composition("H2O"), 300)
    assert freed_entry != FREEDReferenceEntry(Composition("H2O"), 400)
    assert freed_entry != mp_entries[0]


def test_hash(freed_entry):
    assert hash(freed_entry) == hash(FREEDReferenceEntry(Composition("H2O"), 300))
    assert hash(freed_entry) != hash(FREEDReferenceEntry(Composition("H2O"), 400))
