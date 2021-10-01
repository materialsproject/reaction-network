import pytest
from pathlib import Path
from monty.serialization import loadfn
from rxn_network.entries.entry_set import GibbsEntrySet

TEST_FILES_PATH = Path(__file__).parent / "test_files"


def pytest_itemcollected(item):
    """Make tests names more readable in the tests output."""
    item._nodeid = (
        item._nodeid.replace(".py", "")
        .replace("tests/", "")
        .replace("test_", "")
        .replace("_", " ")
        .replace("Test", "")
        .replace("Class", " class")
        .lower()
    )
    doc = item.obj.__doc__.strip() if item.obj.__doc__ else ""
    if doc:
        item._nodeid = item._nodeid.split("::")[0] + "::" + doc


@pytest.fixture(
    params=[
        "Mn-O-Y_entries.json.gz",
    ],
    scope="session",
)
def mp_entries(request):
    mp_entries = loadfn(TEST_FILES_PATH / request.param)
    return mp_entries


@pytest.fixture(scope="session")
def gibbs_entries(mp_entries):
    entries = GibbsEntrySet.from_entries(mp_entries, temperature=1000)
    return entries


@pytest.fixture(scope="session")
def filtered_entries(gibbs_entries):
    filtered_entries = gibbs_entries.filter_by_stability(0.0)
    return filtered_entries


@pytest.fixture(scope="session")
def computed_rxn():
    """ 2 YOCl + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl"""
    return loadfn(TEST_FILES_PATH / "computed_rxn.json.gz")
