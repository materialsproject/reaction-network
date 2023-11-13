""" Tests for ChemicalPotentialDiagram"""
import pytest
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram


@pytest.fixture()
def cpd(gibbs_entries):
    return ChemicalPotentialDiagram(gibbs_entries)


def test_metastable_domains(cpd):
    assert not set(cpd.domains.keys()) & set(cpd.metastable_domains.keys())
