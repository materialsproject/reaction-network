import itertools
import re
import warnings
from copy import deepcopy
from typing import Iterable, List, Optional, Union

from maggma.stores import MongoStore
from pymatgen.core.composition import Element
from pymatgen.core.structure import Structure
from pymatgen.entries import Entry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rxn_network.core.composition import Composition
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.utils.funcs import get_logger

logger = get_logger(__name__)


def process_entries(
    entries: Iterable[Entry],
    temperature: float,
    include_nist_data: bool,
    include_barin_data: bool,
    include_freed_data: bool,
    e_above_hull: float,
    include_polymorphs: bool,
    formulas_to_include: Iterable[str],
    calculate_e_above_hulls: bool = False,
) -> GibbsEntrySet:
    """

    Args:
        entries: Iterable of Entry objects to process.
        temperature (float): Temperature (K) at which to build GibbsComputedEntry
            objects
        include_nist_data (bool): Whether or not to include NIST data when constructing
            the GibbsComputedEntry objects. Defaults to True.
        include_barin_data (bool): Whether or not to include Barin data when
            constructing the GibbsComputedEntry objects. Defaults to False.
        e_above_hull (float): Only include entries with an energy above hull below this
            value (eV)
        include_polymorphs (bool): Whether or not to include metastable polymorphs.
            Defaults to False.
        formulas_to_include: Formulas to ensure are in the entries.

    Returns:
        A GibbsEntrySet object containing GibbsComputedEntry objects with specified
        constraints.
    """
    entry_set = GibbsEntrySet.from_entries(
        entries=entries,
        temperature=temperature,
        include_nist_data=include_nist_data,
        include_barin_data=include_barin_data,
        include_freed_data=include_freed_data,
    )
    entry_set = entry_set.filter_by_stability(
        e_above_hull=e_above_hull, include_polymorphs=include_polymorphs
    )
    if calculate_e_above_hulls:
        entry_set = GibbsEntrySet(deepcopy(entry_set), calculate_e_above_hulls=True)
    included_entries = [initialize_entry(f, entry_set) for f in formulas_to_include]

    entry_set.update(included_entries)

    return entry_set


def initialize_entry(formula: str, entry_set: GibbsEntrySet, stabilize: bool = False):
    """
    Acquire a (stabilized) entry by user-specified formula.

    Args:
        formula: Chemical formula
        entry_set: GibbsEntrySet containing 1 or more entries corresponding to
            given formula
        stabilize: Whether or not to stabilize the entry by decreasing its energy
            such that it is 'on the hull'
    """
    try:
        entry = entry_set.get_min_entry_by_formula(formula)
    except KeyError:
        entry = entry_set.get_interpolated_entry(formula)
        warnings.warn(
            f"Using interpolated entry for {entry.composition.reduced_formula}"
        )

    if stabilize:
        entry = entry_set.get_stabilized_entry(entry)

    return entry


def get_entries(  # noqa: MC0001
    db: MongoStore,
    chemsys_formula_id_criteria: Union[str, dict],
    compatible_only: bool = True,
    inc_structure: Optional[str] = None,
    property_data: Optional[List[str]] = None,
    use_premade_entries: bool = False,
    conventional_unit_cell: bool = False,
    sort_by_e_above_hull: bool = False,
):  # pragma: no cover
    """
    Get a list of ComputedEntries or ComputedStructureEntries corresponding
    to a chemical system, formula, or materials_id or full criteria.

    WARNING:
        This function is legacy code directly adapted from pymatgen.ext.matproj. It is
        not broadly useful or applicable to other databases. It is only used in jobs
        interfaced directly with internal databases at Materials Project. This code is
        not adequately tested and may not work as expected.

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
        criteria = parse_criteria(chemsys_formula_id_criteria)
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
                f"{d['pseudo_potential']['functional']} {label}"
                for label in d["pseudo_potential"].get("labels", [])
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
) -> List[ComputedEntry]:  # noqa: MC0001  # pragma: no cover
    """
    Helper method for getting all entries in a total chemical system by querying
    database for all sub-chemical systems. Code adadpted from pymatgen.ext.matproj
    and modified to support very large chemical systems.

    WARNING:
        This function is legacy code directly adapted from pymatgen.ext.matproj. It is
        not broadly useful or applicable to other databases. It is only used in jobs
        interfaced directly with internal databases at Materials Project. This code is
        not adequately tested and may not work as expected.

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

    def divide_chunks(my_list, n):
        for i in range(0, len(my_list), n):
            yield my_list[i : i + n]

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
    else:  # for very large chemical systems, use a different approach
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


def parse_criteria(criteria_string):  # pragma: no cover
    """
    Parses a powerful and simple string criteria and generates a proper
    mongo syntax criteria.

    Args:
        criteria_string (str): A string representing a search criteria.
            Also supports wild cards. E.g.,
            something like "*2O" gets converted to
            {'pretty_formula': {'$in': [u'B2O', u'Xe2O', u"Li2O", ...]}}

            Other syntax examples:
                mp-1234: Interpreted as a Materials ID.
                Fe2O3 or *2O3: Interpreted as reduced formulas.
                Li-Fe-O or *-Fe-O: Interpreted as chemical systems.

            You can mix and match with spaces, which are interpreted as
            "OR". E.g., "mp-1234 FeO" means query for all compounds with
            reduced formula FeO or with materials_id mp-1234.

    Returns:
        A mongo query dict.
    """
    toks = criteria_string.split()

    def parse_sym(sym):
        if sym == "*":
            return [el.symbol for el in Element]
        m = re.match(r"\{(.*)\}", sym)
        if m:
            return [s.strip() for s in m.group(1).split(",")]
        return [sym]

    def parse_tok(t):
        if re.match(r"\w+-\d+", t):
            return {"task_id": t}
        if "-" in t:
            elements = [parse_sym(sym) for sym in t.split("-")]
            chemsyss = []
            for cs in itertools.product(*elements):
                if len(set(cs)) == len(cs):
                    # Check for valid symbols
                    cs = [Element(s).symbol for s in cs]
                    chemsyss.append("-".join(sorted(cs)))
            return {"chemsys": {"$in": chemsyss}}
        all_formulas = set()
        explicit_els = []
        wild_card_els = []
        for sym in re.findall(r"(\*[\.\d]*|\{.*\}[\.\d]*|[A-Z][a-z]*)[\.\d]*", t):
            if ("*" in sym) or ("{" in sym):
                wild_card_els.append(sym)
            else:
                m = re.match(r"([A-Z][a-z]*)[\.\d]*", sym)
                explicit_els.append(m.group(1))
        nelements = len(wild_card_els) + len(set(explicit_els))
        parts = re.split(r"(\*|\{.*\})", t)
        parts = [parse_sym(s) for s in parts if s != ""]
        for f in itertools.product(*parts):
            c = Composition("".join(f))
            if len(c) == nelements:
                # Check for valid Elements in keys.
                for e in c:
                    Element(e.symbol)
                all_formulas.add(c.reduced_formula)
        return {"pretty_formula": {"$in": list(all_formulas)}}

    if len(toks) == 1:
        return parse_tok(toks[0])
    return {"$or": list(map(parse_tok, toks))}
