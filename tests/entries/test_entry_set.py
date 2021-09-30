""" Tests for GibbsEntrySet. """
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram

from rxn_network.entries.entry_set import GibbsEntrySet


@pytest.mark.parametrize(
    "e_above_hull, expected_phases",
    [
        (
            0.030,
            {
                "Mn",
                "Mn2O3",
                "Mn3O4",
                "Mn5O8",
                "MnO",
                "MnO2",
                "O2",
                "Y",
                "Y2Mn2O7",
                "Y2O3",
                "YMn12",
                "YMn2O5",
                "YMnO3",
            },
        ),
        (
            0.100,
            {
                "YMnO3",
                "Y",
                "Y2O3",
                "Mn5O8",
                "Mn",
                "Y2Mn2O7",
                "YMn2O4",
                "YMn2O5",
                "MnO2",
                "Mn21O40",
                "Mn3O4",
                "Mn7O12",
                "MnO",
                "O2",
                "Mn2O3",
                "YMn12",
            },
        ),
    ],
)
def test_filter_by_stability(e_above_hull, expected_phases, gibbs_entries):
    filtered_entries = gibbs_entries.filter_by_stability(e_above_hull)

    actual_phases = {e.composition.reduced_formula for e in filtered_entries}

    assert actual_phases == expected_phases


def test_build_indices(gibbs_entries):
    entries = gibbs_entries.copy()
    entries.build_indices()

    num_entries = len(entries)

    indices = [e.data["idx"] for e in entries.entries_list]
    assert indices == list(range(num_entries))


def test_get_min_entry_by_formula(gibbs_entries):
    f_id = [("YMnO3", "mp-19385"), ("Mn2O3", "mp-1172875"), ("MnO2", "mp-1279979")]
    for f, entry_id in f_id:
        assert gibbs_entries.get_min_entry_by_formula(f).entry_id == entry_id


def test_stabilize_entry(gibbs_entries):
    entries = gibbs_entries.copy()

    formulas = ["YMn3O6", "Mn2O5"]

    for f in formulas:
        e = entries.get_min_entry_by_formula(f)
        e_stable = entries.stabilize_entry(e)
        entries.add(e_stable)

        assert e_stable in PhaseDiagram(entries).stable_entries


def test_from_pd(mp_entries):
    entries = GibbsEntrySet.from_pd(PhaseDiagram(mp_entries), temperature=1000)
    assert entries is not None
