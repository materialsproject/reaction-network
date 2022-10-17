from pathlib import Path

import pytest
from jobflow.core.store import JobStore
from maggma.stores import MemoryStore
from monty.serialization import loadfn

from rxn_network.core.composition import Composition
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.hull import InterfaceReactionHull

# load files
TEST_FILES_PATH = Path(__file__).parent / "test_files"

MN_O_Y_ENTRIES = loadfn(TEST_FILES_PATH / "Mn_O_Y_entries.json.gz")
CL_MN_NA_O_Y_ENTRIES = loadfn(TEST_FILES_PATH / "Cl_Mn_Na_O_Y_entries.json.gz")
YMNO3_RXNS = loadfn(TEST_FILES_PATH / "ymno3_rxns.json.gz")
BAO_TIO2_RXNS = loadfn(TEST_FILES_PATH / "bao_tio2_rxns.json.gz")
COMPUTED_RXN = loadfn(TEST_FILES_PATH / "computed_rxn.json.gz")
ALL_YMNO_RXNS = loadfn(TEST_FILES_PATH / "all_ymno_rxns.json.gz")
YMN2O5_MN3O4_PATHS = loadfn(TEST_FILES_PATH / "ymn2o5_mn3o4_paths.json.gz")
YMNO_RN = loadfn(TEST_FILES_PATH / "ymno_rn.json.gz")


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
def entries():
    return GibbsEntrySet(CL_MN_NA_O_Y_ENTRIES)


@pytest.fixture(scope="session")
def filtered_entries():
    filtered_entries = GibbsEntrySet.from_entries(
        MN_O_Y_ENTRIES, temperature=1000, include_barin_data=False
    ).filter_by_stability(0.0)
    return filtered_entries


@pytest.fixture(scope="session")
def computed_rxn():
    """2 YOCl + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl"""
    return COMPUTED_RXN


@pytest.fixture(scope="session")
def ymno3_rxns():
    return YMNO3_RXNS


@pytest.fixture(scope="session")
def all_ymno_rxns():
    return ALL_YMNO_RXNS


@pytest.fixture(scope="session")
def bao_tio2_rxns():
    return BAO_TIO2_RXNS


@pytest.fixture(scope="session")
def irh_batio(bao_tio2_rxns):
    return InterfaceReactionHull(
        c1=Composition("BaO"), c2=Composition("TiO2"), reactions=bao_tio2_rxns
    )


@pytest.fixture(scope="session")
def ymn2o5_mn3o4_paths():
    return YMN2O5_MN3O4_PATHS


@pytest.fixture(scope="session")
def ymno_rn():
    rn = YMNO_RN
    rn.load_graph(str(TEST_FILES_PATH / "ymno_rn_graph.gt.gz"))
    return rn


@pytest.fixture(scope="session")
def job_store():
    additional_stores = {"rxns": MemoryStore(), "entries": MemoryStore()}
    return JobStore(MemoryStore(), additional_stores=additional_stores)
