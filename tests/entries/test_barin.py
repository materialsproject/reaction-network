"""Tests for BarinReferenceEntry"""

import pytest

from rxn_network.core.composition import Composition
from rxn_network.entries.barin import BarinReferenceEntry


@pytest.fixture
def barin_entry():
    return BarinReferenceEntry(Composition("H2O"), 300)


def test_repr(barin_entry):
    assert (
        repr(barin_entry) == "BarinReferenceEntry | H2O\nGibbs Energy (300 K) = -2.3686"
    )
