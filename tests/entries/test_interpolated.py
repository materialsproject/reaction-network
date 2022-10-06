"""Tests for interpolated entry"""

import pytest
from pymatgen.core.composition import Element

from rxn_network.entries.interpolated import InterpolatedEntry


@pytest.fixture
def entry():
    """Create entry"""
    entry = InterpolatedEntry(
        composition="H3O",
        energy=-1.0,
    )
    return entry


def test_to_grand_entry(entry):
    """Test to_grand_entry"""
    grand_entry = entry.to_grand_entry({Element("O"): 0.0})
    assert grand_entry.__class__.__name__ == "GrandPotPDEntry"
    assert grand_entry.energy == entry.energy
    assert grand_entry.energy_per_atom != entry.energy_per_atom
    assert grand_entry.composition != entry.composition


def test_is_experimental(entry):
    """Test is_experimental"""
    assert entry.is_experimental is False
