from pathlib import Path

import pytest
from jobflow.core.store import JobStore
from maggma.stores import MemoryStore
from monty.serialization import loadfn
from rxn_network.core import Composition
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.entries.interpolated import InterpolatedEntry
from rxn_network.reactions.hull import InterfaceReactionHull

# load files
TEST_FILES_PATH = Path(__file__).parent / "test_files"
ENTRIES_LIST = loadfn(TEST_FILES_PATH / "Mn_O_Y_entries.json.gz")
ENTRIES_WITH_NA_CL = loadfn(TEST_FILES_PATH / "Cl_Mn_Na_O_Y_entries.json.gz")


@pytest.fixture(scope="session")
def mp_entries():
    return ENTRIES_LIST


@pytest.fixture(scope="session")
def gibbs_entries():
    return GibbsEntrySet.from_computed_entries(
        ENTRIES_LIST,
        temperature=1000,
    )


@pytest.fixture(scope="session")
def entries():
    """Doesn't apply Gibbs corrections"""
    return GibbsEntrySet(ENTRIES_WITH_NA_CL)


@pytest.fixture(scope="session")
def gibbs_entries_with_na_cl(entries):
    return GibbsEntrySet.from_computed_entries(list(entries), temperature=1000).filter_by_stability(0.0)


@pytest.fixture(scope="session")
def filtered_entries():
    return GibbsEntrySet.from_computed_entries(
        ENTRIES_LIST,
        temperature=1000,
    ).filter_by_stability(0.0)


@pytest.fixture()
def interpolated_entry():
    """Create entry"""
    return InterpolatedEntry(
        composition="Y3O8",
        energy=-1.0,
    )


@pytest.fixture(scope="session")
def computed_rxn():
    """2 YOCl + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl"""
    return loadfn(TEST_FILES_PATH / "computed_rxn.json.gz")


@pytest.fixture(scope="session")
def ymno3_rxns():
    return loadfn(TEST_FILES_PATH / "ymno3_rxns.json.gz")


@pytest.fixture(scope="session")
def ymno3_gibbs_rxns():
    return loadfn(TEST_FILES_PATH / "ymno3_gibbs_rxns.json.gz")


@pytest.fixture(scope="session")
def all_ymno_rxns():
    return loadfn(TEST_FILES_PATH / "all_ymno_rxns.json.gz")


@pytest.fixture(scope="session")
def bao_tio2_rxns():
    return loadfn(TEST_FILES_PATH / "bao_tio2_rxns.json.gz")


@pytest.fixture(scope="session")
def irh_batio(bao_tio2_rxns):
    return InterfaceReactionHull(c1=Composition("BaO"), c2=Composition("TiO2"), reactions=bao_tio2_rxns)


@pytest.fixture(scope="session")
def ymn2o5_mn3o4_paths():
    return loadfn(TEST_FILES_PATH / "ymn2o5_mn3o4_network_paths.json.gz")


@pytest.fixture(scope="session")
def mn_o_y_network_entries():
    return loadfn(TEST_FILES_PATH / "Mn_O_Y_network_entries.json.gz")


@pytest.fixture(scope="session")
def ymno_rn():
    return loadfn(TEST_FILES_PATH / "ymno_rn.json.gz")


@pytest.fixture(scope="session")
def job_store():
    additional_stores = {
        "rxns": MemoryStore(),
        "entries": MemoryStore(),
        "network": MemoryStore(),
        "paths": MemoryStore(),
    }
    return JobStore(MemoryStore(), additional_stores=additional_stores)
