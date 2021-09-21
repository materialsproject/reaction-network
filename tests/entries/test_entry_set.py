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
                "Mn-O-Y": {
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
                "Fe-Li-O-P": {
                    "Li3PO4",
                    "Fe2PO5",
                    "Fe7(P2O7)4",
                    "LiFe(PO3)4",
                    "FePO4",
                    "Fe(PO3)3",
                    "Fe(PO3)2",
                    "LiFePO4",
                    "LiFeP",
                    "LiFe(PO3)3",
                    "P4O9",
                    "FeP4",
                    "LiP",
                    "Fe3(PO4)2",
                    "Li2Fe(PO3)4",
                    "Li2O",
                    "FeP4O11",
                    "P2O5",
                    "P",
                    "Li5FeO4",
                    "Li2FeP2O7",
                    "LiFeO2",
                    "Fe4P2O9",
                    "LiFe2P5O16",
                    "Fe2O3",
                    "Li2Fe3(P2O7)2",
                    "Li4Fe(PO4)2",
                    "Fe2P",
                    "LiPO3",
                    "Fe3P",
                    "Li4Fe2O5",
                    "Fe2P2O7",
                    "Li9Fe3P8O29",
                    "FeO",
                    "Li2FeO2",
                    "Fe",
                    "Fe2(PO3)5",
                    "FeP2",
                    "LiFeP2O7",
                    "LiFe2P3O10",
                    "Fe2P5O16",
                    "Fe3(P2O7)2",
                    "FeP",
                    "Fe3(P3O10)2",
                    "PO2",
                    "Li3P",
                    "Li",
                    "Li21(FeO4)4",
                    "Fe3O4",
                    "Li4P2O7",
                    "Li3Fe2P5O16",
                    "Li3P7",
                    "O2",
                },
            },
        ),
        (
            0.100,
            {
                "Mn-O-Y": {
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
                "Fe-Li-O-P": {
                    "Li2FeP4O13",
                    "LiFe4(PO4)3",
                    "P2O3",
                    "Li3(FeO2)4",
                    "Li8FeO6",
                    "Fe(PO3)5",
                    "Fe23O25",
                    "P",
                    "Li5FeO4",
                    "LiFeO2",
                    "LiFe5P3O13",
                    "LiFe2P5O16",
                    "FeP2O7",
                    "Li9Fe5O12",
                    "Li5Fe(P2O7)2",
                    "Li5Fe4(PO4)4",
                    "Fe2O3",
                    "Li2FeP3O10",
                    "Li2Fe3(P2O7)2",
                    "Li4Fe(PO3)6",
                    "Fe3P",
                    "Li3Fe5(PO4)6",
                    "Li7Fe2P7O24",
                    "Fe13O15",
                    "Fe2(PO3)5",
                    "Fe14O15",
                    "Li5(FeO2)4",
                    "LiFe4P7O24",
                    "LiFeP2O7",
                    "Li9Fe23O32",
                    "Li35(FeO4)8",
                    "Fe2P5O16",
                    "Fe3(P2O7)2",
                    "Li4FeP2O9",
                    "Li14Fe4O13",
                    "Li8Fe7O15",
                    "Fe(PO3)4",
                    "Fe3P3O11",
                    "Li7Fe4(PO4)6",
                    "Fe4O5",
                    "PO2",
                    "Li3Fe2P5O16",
                    "Li2(FeO2)3",
                    "Fe5(PO4)4",
                    "O2",
                    "Fe(PO3)2",
                    "Li2Fe3P5O18",
                    "Fe(PO3)3",
                    "LiFePO4",
                    "LiFe4(PO4)4",
                    "Fe2P3O10",
                    "Li6Fe5O12",
                    "P4O7",
                    "Fe12O13",
                    "FeP4",
                    "Fe17O18",
                    "LiFe2(PO3)5",
                    "Li5Fe2P5O18",
                    "Li2O",
                    "Li8Fe7(PO4)8",
                    "Li7Fe4(P2O7)4",
                    "P2O5",
                    "Li2FeP2O7",
                    "Fe41O56",
                    "Li3Fe2(PO3)7",
                    "Fe38O39",
                    "Fe5(P2O7)4",
                    "Li7Fe3O8",
                    "LiPO3",
                    "LiFe3(P3O10)2",
                    "FeO",
                    "Li7Fe5O12",
                    "Li2Fe5(PO4)4",
                    "Fe11O12",
                    "LiFe2(PO4)2",
                    "Fe5(P3O11)2",
                    "Fe9O10",
                    "Fe21O23",
                    "FeP2",
                    "Fe3P4O15",
                    "LiFe2P3O10",
                    "Li3FePO5",
                    "Li5Fe5(PO4)6",
                    "Fe13O14",
                    "LiFe23O32",
                    "Li3Fe(PO4)2",
                    "FeP",
                    "Fe5P3O13",
                    "Li21(FeO4)4",
                    "Li",
                    "Li4P2O7",
                    "LiFe2O3",
                    "Li3PO4",
                    "Li4Fe7(PO4)6",
                    "P4O9",
                    "FePO4",
                    "LiFe(PO3)5",
                    "LiFe(PO3)3",
                    "LiFeP",
                    "Li(Fe3P2)2",
                    "Li9(FeO4)2",
                    "Li2Fe(PO3)4",
                    "Li7Fe4P9O32",
                    "Li2FeO3",
                    "FeP4O11",
                    "Fe3PO7",
                    "Fe4P2O9",
                    "LiFe3O4",
                    "Fe15O16",
                    "Li4Fe(PO4)2",
                    "Fe2P",
                    "LiFe5(P2O7)4",
                    "Li5Fe3O8",
                    "LiFe2P3O11",
                    "Li4Fe2O5",
                    "Li4Fe2(PO4)3",
                    "Li9Fe3P8O29",
                    "Li(Fe2O3)4",
                    "LiP5",
                    "LiP7",
                    "Li3Fe2(P2O7)2",
                    "Fe3(P3O10)2",
                    "Li3P",
                    "Li4(FeO2)5",
                    "Fe3O4",
                    "Li2Fe3P9O28",
                    "Fe7O8",
                    "Li2Fe12P7",
                    "Fe7(PO4)6",
                    "Fe10O11",
                    "Li6Fe9(PO4)8",
                    "Li3Fe5O8",
                    "Fe2PO5",
                    "Fe7(P2O7)4",
                    "LiFe(PO3)4",
                    "Fe4(P2O7)3",
                    "LiP",
                    "Fe8O9",
                    "Fe3(PO4)2",
                    "Li5Fe7O12",
                    "Li9Fe4(PO5)4",
                    "Li8(FeO2)5",
                    "Li2FePO5",
                    "Li9Fe7(PO4)12",
                    "Fe4(PO4)3",
                    "Li2Fe(PO3)5",
                    "Li3FeO3",
                    "LiFe3P4O15",
                    "Li4FeO3",
                    "Li5Fe2(PO4)3",
                    "Li4Fe7O12",
                    "Li4Fe5(P3O11)2",
                    "Li8Fe2O9",
                    "Fe2P2O7",
                    "Li2FeO2",
                    "Fe",
                    "Li5Fe4(P2O7)4",
                    "LiFe3P3O11",
                    "Li6FeO4",
                    "Li6Fe3P8O29",
                    "Li8Fe2O7",
                    "Li2Fe3(PO4)3",
                    "Li2Fe2(PO4)3",
                    "Li(FeP)2",
                    "Fe32O35",
                    "Fe4P7O24",
                    "Li3Fe5O9",
                    "LiFe5O8",
                    "Li5FeP3O11",
                    "Li3P7",
                    "Li(FeO2)2",
                    "Fe7(PO5)4",
                    "Li4Fe3P4O15",
                    "Li5Fe11O16",
                    "Li3Fe2(PO4)3",
                    "Li3Fe3(PO4)4",
                },
            },
        ),
    ],
)
def test_filter_by_stability(e_above_hull, expected_phases, gibbs_entries):
    chemsys = "-".join(sorted(gibbs_entries.chemsys))
    filtered_entries = gibbs_entries.filter_by_stability(e_above_hull)

    actual_phases = None
    if chemsys == "Mn-O-Y" or chemsys == "Fe-Li-O-P":
        actual_phases = {e.composition.reduced_formula for e in filtered_entries}

    assert actual_phases == expected_phases[chemsys]


def test_build_indices(gibbs_entries):
    entries = gibbs_entries.copy()
    entries.build_indices()

    num_entries = len(entries)

    indices = [e.data["idx"] for e in entries.entries_list]
    assert indices == list(range(num_entries))


def test_get_min_entry_by_formula(gibbs_entries):
    chemsys = "-".join(sorted(gibbs_entries.chemsys))

    f_id = [("YMnO3", "mp-19385"), ("Mn2O3", "mp-1172875"), ("MnO2", "mp-1279979")]
    f_id2 = [
        ("LiFePO4", "mp-756958"),
        ("Fe2O3", "mp-19770"),
        ("Li2O", "NISTReferenceEntry"),
    ]

    test_formulas = None
    if chemsys == "Mn-O-Y":
        test_formulas = f_id
    elif chemsys == "Fe-Li-O-P":
        test_formulas = f_id2

    for f, entry_id in test_formulas:
        assert gibbs_entries.get_min_entry_by_formula(f).entry_id == entry_id


def test_stabilize_entry(gibbs_entries):
    chemsys = "-".join(sorted(gibbs_entries.chemsys))
    entries = gibbs_entries.copy()

    f1 = ["YMn3O6", "Mn2O5"]
    f2 = ["LiP", "LiFeO3"]

    test_formulas = None
    if chemsys == "Mn-O-Y":
        test_formulas = f1
    elif chemsys == "Fe-Li-O-P":
        test_formulas = f2

    for f in test_formulas:
        e = entries.get_min_entry_by_formula(f)
        e_stable = entries.stabilize_entry(e)
        entries.add(e_stable)

        assert e_stable in PhaseDiagram(entries).stable_entries


def test_from_pd(mp_entries):
    entries = GibbsEntrySet.from_pd(PhaseDiagram(mp_entries), temperature=1000)
    assert entries is not None
