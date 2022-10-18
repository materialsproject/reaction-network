"""Tests for interpolated entry"""

from pymatgen.core.composition import Element


def test_to_grand_entry(interpolated_entry):
    """Test to_grand_entry"""
    grand_entry = interpolated_entry.to_grand_entry({Element("O"): 0.0})
    assert grand_entry.__class__.__name__ == "GrandPotPDEntry"
    assert grand_entry.energy == interpolated_entry.energy
    assert grand_entry.energy_per_atom != interpolated_entry.energy_per_atom
    assert grand_entry.composition != interpolated_entry.composition


def test_is_experimental(interpolated_entry):
    """Test is_experimental"""
    assert interpolated_entry.is_experimental is False
