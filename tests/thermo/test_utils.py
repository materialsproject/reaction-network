"""Tests for thermo/utils.py"""

from rxn_network.thermo.utils import expand_pd


def test_expand_pd(entries):
    """Test for expand_pd"""
    pd_dict = expand_pd(entries)

    all_elems = {elem for k in pd_dict for elem in k.split("-")}
    assert all_elems == entries.chemsys

    assert set(pd_dict.keys()) == {  # this might change, but good to check
        "Mn-Na-O",
        "Cl-O-Y",
        "Cl-Mn-O",
        "Mn-O-Y",
        "Cl-Na-Y",
        "Cl-Na-O",
        "Cl-Mn-Na",
        "Na-O-Y",
    }
