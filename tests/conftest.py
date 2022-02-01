import pytest
from pathlib import Path
from monty.serialization import loadfn
from rxn_network.entries.entry_set import GibbsEntrySet

TEST_FILES_PATH = Path(__file__).parent / "test_files"

MN_O_Y_ENTRIES = loadfn(TEST_FILES_PATH / "Mn_O_Y_entries.json.gz")
CL_MN_NA_O_Y_ENTRIES = loadfn(TEST_FILES_PATH / "Cl_Mn_Na_O_Y_entries.json.gz")


@pytest.fixture(scope="session")
def entries():
    return GibbsEntrySet(CL_MN_NA_O_Y_ENTRIES)


@pytest.fixture(scope="session")
def mp_entries():
    return MN_O_Y_ENTRIES


@pytest.fixture(scope="session")
def gibbs_entries():
    ents = GibbsEntrySet.from_entries(
        MN_O_Y_ENTRIES, temperature=1000, include_barin_data=False
    )
    return ents


@pytest.fixture(scope="session")
def filtered_entries(gibbs_entries):
    filtered_entries = gibbs_entries.filter_by_stability(0.0)
    return filtered_entries


@pytest.fixture(scope="session")
def computed_rxn():
    """2 YOCl + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl"""
    return loadfn(TEST_FILES_PATH / "computed_rxn.json.gz")


# def pytest_itemcollected(item):
#     """Make tests names more readable in the tests output."""
#     item._nodeid = (
#         item._nodeid.replace(".py", "")
#         .replace("tests/", "")
#         .replace("test_", "")
#         .replace("Test", "")
#         .replace("Class", " class")
#         .lower()
#     )
#     doc = item.obj.__doc__.strip() if item.obj.__doc__ else ""
#     if doc:
#         item._nodeid = item._nodeid.split("::")[0] + "::" + doc
