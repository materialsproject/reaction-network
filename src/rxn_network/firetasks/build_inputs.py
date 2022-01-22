"""
Firetasks for acquiring ComputedEntry data from MPRester or another materials MongoDB
"""
import itertools
import warnings
from typing import Iterable, List, Optional, Union

from fireworks import FiretaskBase, FWAction, explicit_serialize
from maggma.stores import MongoStore
from pymatgen.core.structure import Structure
from pymatgen.entries import Entry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.firetasks.utils import env_chk, get_logger

logger = get_logger(__name__)


@explicit_serialize
class EntriesFromMPRester(FiretaskBase):
    """
    Acquire ComputedStructureEntry objects from the Materials Project database.
    Automatically builds GibbsComputedEntry objects at specified temperature and
    filters by e_above_hull stability.

    Note: the PMG_MAPI_KEY enviornment variable must be properly configured on the
    computing resource.

    Note: This task is not yet compatible with the new Materials Project API.

    Required params:
        chemsys (str): String of chemical system to query (e.g. "Li-Mn-O")
        temperature (float): Temperature (K) at which to build GibbsComputedEntry objects
        e_above_hull (float): Only include entries with an energy above hull below this value (eV)

    Optional params:
        include_polymorphs (bool): Whether or not to include metastable polymorphs.
            Defaults to False.
        include_nist_data (bool): Whether or not to include NIST data when constructing
            the GibbsComputedEntry objects. Defaults to True.
        include_barin_data (bool): Whether or not to include Barin data when constructing
            the GibbsComputedEntry objects. Defaults to False.

    """

    required_params = ["chemsys", "temperature", "e_above_hull"]
    optional_params = [
        "include_nist_data",
        "include_barin_data",
        "include_polymorphs",
    ]

    def run_task(self, fw_spec):
        chemsys = self["chemsys"]
        temperature = self["temperature"]
        e_above_hull = self["e_above_hull"]
        include_nist_data = self.get("include_nist_data", True)
        include_barin_data = self.get("include_barin_data", False)
        include_polymorphs = self.get("include_polymorphs", False)

        with MPRester() as mpr:
            entries = mpr.get_entries_in_chemsys(
                elements=chemsys, inc_structure="final"
            )

        entries = process_entries(
            entries,
            temperature=temperature,
            include_nist_data=include_nist_data,
            include_barin_data=include_barin_data,
            e_above_hull=e_above_hull,
            include_polymorphs=include_polymorphs,
        )
        return FWAction(update_spec={"entries": entries})


@explicit_serialize
class EntriesFromDb(FiretaskBase):
    """
    Acquire ComputedStructureEntry objects from a custom materials MongoDB.
    Automatically builds GibbsComputedEntry objects at specified temperature and
    filters by e_above_hull stability.

    Required params:
        entry_db_file (str): path to db.json file with MongoDB credentials.
        chemsys (str): String of chemical system to query (e.g. "Li-Mn-O")
        temperature (float): Temperature (K) at which to build GibbsComputedEntry objects
        e_above_hull (float): Only include entries with an energy above hull below this value (eV)

    Optional params:
        include_polymorphs (bool): Whether or not to include metastable polymorphs.
            Defaults to False.
        include_nist_data (bool): Whether or not to include NIST data when constructing
            the GibbsComputedEntry objects. Defaults to True.
        include_barin_data (bool): Whether or not to include Barin data when constructing
            the GibbsComputedEntry objects. Defaults to False.
        inc_structure (str): What type of structure object to include. Defaults to "final".
        compatible_only (bool): Include only entries that are MP compatible. Defaults to True.
        property_data (dict): Optional dictionary of properties to include in each entry.

    """

    required_params = [
        "entry_db_file",
        "chemsys",
        "temperature",
        "e_above_hull",
    ]
    optional_params = [
        "include_nist_data",
        "include_barin_data",
        "include_polymorphs",
        "inc_structure",
        "compatible_only",
        "property_data",
    ]

    def run_task(self, fw_spec):
        db_file = env_chk(self["entry_db_file"], fw_spec)
        chemsys = self["chemsys"]
        temperature = self["temperature"]
        include_nist_data = self.get("include_nist_data", True)
        include_barin_data = self.get("include_barin_data", False)
        e_above_hull = self["e_above_hull"]
        include_polymorphs = self.get("include_polymorphs", False)
        inc_structure = self.get("inc_structure", "final")
        compatible_only = self.get("compatible_only", True)
        property_data = self.get("property_data", None)

        with MongoStore.from_db_file(db_file) as db:
            entries = get_all_entries_in_chemsys(
                db,
                chemsys,
                inc_structure=inc_structure,
                compatible_only=compatible_only,
                property_data=property_data,
                use_premade_entries=False,
            )

        entries = process_entries(
            entries,
            temperature=temperature,
            include_nist_data=include_nist_data,
            include_barin_data=include_barin_data,
            e_above_hull=e_above_hull,
            include_polymorphs=include_polymorphs,
        )
        return FWAction(update_spec={"entries": entries})


def process_entries(
    entries: Iterable[Entry],
    temperature: float,
    include_nist_data: bool,
    include_barin_data: bool,
    e_above_hull: float,
    include_polymorphs: bool,
) -> GibbsEntrySet:
    """

    Args:
        entries: Iterable of Entry objects to process.
        temperature (float): Temperature (K) at which to build GibbsComputedEntry
            objects
        include_nist_data (bool): Whether or not to include NIST data when constructing
            the GibbsComputedEntry objects. Defaults to True.
        include_barin_data (bool): Whether or not to include Barin data when constructing
            the GibbsComputedEntry objects. Defaults to False.
        e_above_hull (float): Only include entries with an energy above hull below this value (eV)
        include_polymorphs (bool): Whether or not to include metastable polymorphs.
            Defaults to False.

    Returns:
        A GibbsEntrySet object containing GibbsComputedEntry objects with specified constraints.
    """
    entry_set = GibbsEntrySet.from_entries(
        entries=entries,
        temperature=temperature,
        include_nist_data=include_nist_data,
        include_barin_data=include_barin_data,
    )
    entry_set = entry_set.filter_by_stability(
        e_above_hull=e_above_hull, include_polymorphs=include_polymorphs
    )
    return entry_set


def get_entries(  # noqa: C901
    db: MongoStore,
    chemsys_formula_id_criteria: Union[str, dict],
    compatible_only: bool = True,
    inc_structure: Optional[str] = None,
    property_data: Optional[List[str]] = None,
    use_premade_entries: bool = False,
    conventional_unit_cell: bool = False,
    sort_by_e_above_hull: bool = False,
):
    """
    Get a list of ComputedEntries or ComputedStructureEntries corresponding
    to a chemical system, formula, or materials_id or full criteria. Code adapted
    from pymatgen.ext.matproj.

    Args:
        db: MongoStore object with database connection
        chemsys_formula_id_criteria: A chemical system
            (e.g., Li-Fe-O), or formula (e.g., Fe2O3) or materials_id
            (e.g., mp-1234) or full Mongo-style dict criteria.
        compatible_only: Whether to return only "compatible"
            entries. Compatible entries are entries that have been
            processed using the MaterialsProjectCompatibility class,
            which performs adjustments to allow mixing of GGA and GGA+U
            calculations for more accurate phase diagrams and reaction
            energies.
        inc_structure: If None, entries returned are
            ComputedEntries. If inc_structure="initial",
            ComputedStructureEntries with initial structures are returned.
            Otherwise, ComputedStructureEntries with final structures
            are returned.
        property_data: Specify additional properties to include in
            entry.data. If None, no data. Should be a subset of
            supported_properties.
        use_premade_entries: Whether to use entry objects that have already been
            constructed. Defaults to False.
        conventional_unit_cell: Whether to get the standard
            conventional unit cell
        sort_by_e_above_hull: Whether to sort the list of entries by
            e_above_hull (will query e_above_hull as a property_data if True).
    Returns:
        List of ComputedEntry or ComputedStructureEntry objects.
    """
    params = [
        "deprecated",
        "run_type",
        "is_hubbard",
        "pseudo_potential",
        "hubbards",
        "potcar_symbols",
        "oxide_type",
    ]
    props = ["final_energy", "unit_cell_formula", "task_id"] + params
    if sort_by_e_above_hull:
        if property_data and "e_above_hull" not in property_data:
            property_data.append("e_above_hull")
        elif not property_data:
            property_data = ["e_above_hull"]
    if property_data:
        props += property_data
    if inc_structure:
        if inc_structure == "initial":
            props.append("initial_structure")
        else:
            props.append("structure")

    if not isinstance(chemsys_formula_id_criteria, dict):
        criteria = MPRester.parse_criteria(chemsys_formula_id_criteria)
    else:
        criteria = chemsys_formula_id_criteria

    if use_premade_entries:
        props = ["entries", "deprecated"]

    entries = []
    for d in db.query(criteria, props):
        if d.get("deprecated"):
            continue
        if use_premade_entries:
            ent = d["entries"]
            if ent.get("GGA"):
                e = ComputedStructureEntry.from_dict(ent["GGA"])
            elif ent.get("GGA+U"):
                e = ComputedStructureEntry.from_dict(ent["GGA+U"])
            else:
                print(f"Missing entry for {d['_id']}")
                continue
        else:
            d["potcar_symbols"] = [
                f"{d['pseudo_potential']['functional']} {l}"
                for l in d["pseudo_potential"].get("labels", [])
            ]
            data = {"oxide_type": d["oxide_type"]}
            if property_data:
                data.update({k: d[k] for k in property_data})
            if not inc_structure:
                e = ComputedEntry(
                    d["unit_cell_formula"],
                    d["final_energy"],
                    parameters={k: d[k] for k in params},
                    data=data,
                    entry_id=d["task_id"],
                )
            else:
                prim = Structure.from_dict(
                    d["initial_structure"]
                    if inc_structure == "initial"
                    else d["structure"]
                )
                if conventional_unit_cell:
                    s = SpacegroupAnalyzer(prim).get_conventional_standard_structure()
                    energy = d["final_energy"] * (len(s) / len(prim))
                else:
                    s = prim.copy()
                    energy = d["final_energy"]
                e = ComputedStructureEntry(
                    s,
                    energy,
                    parameters={k: d[k] for k in params},
                    data=data,
                    entry_id=d["task_id"],
                )
        entries.append(e)

    if compatible_only:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Failed to guess oxidation states.*"
            )
            entries = MaterialsProject2020Compatibility().process_entries(
                entries, clean=True
            )

    if sort_by_e_above_hull:
        entries = sorted(entries, key=lambda entry: entry.data["e_above_hull"])

    return entries


def get_all_entries_in_chemsys(
    db: MongoStore,
    elements: Union[str, List[str]],
    compatible_only: bool = True,
    inc_structure: Optional[str] = None,
    property_data: Optional[list] = None,
    use_premade_entries: bool = False,
    conventional_unit_cell: bool = False,
    n: int = 1000,
) -> List[ComputedEntry]:
    """
    Helper method for getting all entries in a total chemical system by querying
    database for all sub-chemical systems. Code adadpted from pymatgen.ext.matproj
    and modified to support very large chemical systems.

    Args:
        db: MongoStore object with database connection
        elements (str or [str]): Chemical system string comprising element
            symbols separated by dashes, e.g., "Li-Fe-O" or List of element
            symbols, e.g., ["Li", "Fe", "O"].
        compatible_only (bool): Whether to return only "compatible"
            entries. Compatible entries are entries that have been
            processed using the MaterialsProjectCompatibility class,
            which performs adjustments to allow mixing of GGA and GGA+U
            calculations for more accurate phase diagrams and reaction
            energies.
        inc_structure (str): If None, entries returned are
            ComputedEntries. If inc_structure="final",
            ComputedStructureEntries with final structures are returned.
            Otherwise, ComputedStructureEntries with initial structures
            are returned.
        property_data (list): Specify additional properties to include in
            entry.data. If None, no data. Should be a subset of
            supported_properties.
        use_premade_entries: Whether to use entry objects that have already been
            constructed. Defaults to False.
        conventional_unit_cell (bool): Whether to get the standard
            conventional unit cell
        n (int): Chunk size, i.e., number of sub-chemical systems to consider
    Returns:
        List of ComputedEntries.
    """

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    if isinstance(elements, str):
        elements = elements.split("-")

    if len(elements) < 13:
        all_chemsyses = []
        for i in range(len(elements)):
            for els in itertools.combinations(elements, i + 1):
                all_chemsyses.append("-".join(sorted(els)))

        all_chemsyses = list(divide_chunks(all_chemsyses, n))

        entries = []
        for chemsys_group in all_chemsyses:
            entries.extend(
                get_entries(
                    db,
                    {"chemsys": {"$in": chemsys_group}},
                    compatible_only=compatible_only,
                    inc_structure=inc_structure,
                    property_data=property_data,
                    use_premade_entries=use_premade_entries,
                    conventional_unit_cell=conventional_unit_cell,
                )
            )
    else:
        entries = get_entries(
            db,
            {"elements": {"$not": {"$elemMatch": {"$nin": elements}}}},
            compatible_only=compatible_only,
            inc_structure=inc_structure,
            property_data=property_data,
            use_premade_entries=use_premade_entries,
            conventional_unit_cell=conventional_unit_cell,
        )

    return entries


def get_entry_task(
    chemsys: Union[str, Iterable[str]],
    entry_set_params: Optional[dict],
    entry_db_file: str,
) -> Union[EntriesFromDb, EntriesFromMPRester]:
    """
    Helper method for choosing an entry-acquiring FireTask depending on what is provided
    by the user. If an entry_db_file is provided, uses the EntriesFromDb task;
    otherwise, uses the EntriesFromMPRester task.

    Args:
        chemsys (str): String of chemical system to query (e.g. "Li-Mn-O")
        entry_set_params (dict): Dictionary of parameters for constructing a
            GibbsEntrySet.
                - temperature (float): Temperature in Kelvin
                - e_above_hull (float): E-above-hull in eV
                - include_nist_data (bool): Whether to include NIST data
                - include_barin_data (bool): Whether to include Barin data
                - include_polymorphs (bool): Whether to include metastable polymorphs
        entry_db_file (str): Path to file containing entries to use.

    Retruns:
    """
    entry_set_params = entry_set_params if entry_set_params else {}
    temperature = entry_set_params.get("temperature", 300)
    e_above_hull = entry_set_params.get("e_above_hull", 0.0)
    include_nist_data = entry_set_params.get("include_nist_data", True)
    include_barin_data = entry_set_params.get("include_barin_data", False)
    include_polymorphs = entry_set_params.get("include_polymorphs", False)

    if bool(entry_db_file):
        entry_task = EntriesFromDb(
            entry_db_file=entry_db_file,
            chemsys=chemsys,
            temperature=temperature,
            include_nist_data=include_nist_data,
            include_barin_data=include_barin_data,
            e_above_hull=e_above_hull,
            include_polymorphs=include_polymorphs,
        )
    else:
        entry_task = EntriesFromMPRester(
            chemsys=chemsys,
            temperature=temperature,
            include_nist_data=include_nist_data,
            include_barin_data=include_barin_data,
            e_above_hull=e_above_hull,
            include_polymorphs=include_polymorphs,
        )

    return entry_task
