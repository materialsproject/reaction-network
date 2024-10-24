"""Utility functions for acquiring, processing, or modifiying entries."""

from __future__ import annotations

import itertools
import re
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

from pymatgen.core.composition import Element
from pymatgen.core.structure import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rxn_network.core import Composition
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.utils.funcs import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from maggma.stores import MongoStore

logger = get_logger(__name__)


def process_entries(
    entries: Iterable[ComputedStructureEntry],
    temperature: float,
    e_above_hull: float,
    filter_at_temperature: int | None = None,
    include_nist_data: bool = True,
    include_freed_data: bool = False,
    include_polymorphs: bool = False,
    include_icsd_entropy: bool = True,
    formulas_to_include: Iterable[str] | None = None,
    calculate_e_above_hulls: bool = False,
    ignore_nist_solids: bool = True,
) -> GibbsEntrySet:
    """Convenience function for processing a set of ComputedStructureEntry objects into a
    GibbsEntrySet with specified parameters. This is used when building entries in most
    of the jobs/flows.

    Args:
        entries: Iterable of ComputedStructureEntry objects. These can be downloaded
            from The Materials Project API or created manually with pymatgen.
        temperature: Temperature [K] for determining Gibbs Free Energy of
            formation, dGf(T).
        e_above_hull: Energy above hull (eV/atom) for thermodynamic stability threshold;
            i.e., include all entries with energies below this value.
        filter_at_temperature: Temperature (in Kelvin) at which entries are filtered for
            thermodynamic stability (e.g., room temperature). Generally, this often
            differs from the synthesis temperature.
        include_nist_data: Whether to include NIST-JANAF data in the entry set.
            Defaults to True.
        include_freed_data: Whether to include FREED data in the entry set. Defaults
            to False. WARNING: This dataset has not been thoroughly tested. Use at
            your own risk!
        include_polymorphs: Whether to include non-ground state polymorphs in the entry
            set. Defaults to False.
        formulas_to_include: An iterable of compositional formulas to ensure are
            included in the processed dataset. Sometimes, entries are filtered out that
            one would like to include, or entries don't exist for those compositions.
        calculate_e_above_hulls: Whether to calculate e_above_hull and store as an
            attribute in the data dictionary for each entry.
        ignore_nist_solids: Whether to ignore NIST data for solids with high melting
            points (Tm >= 1500 ºC). Defaults to True.

    Returns:
        A GibbsEntrySet object containing entry objects with the user-specified
        constraints.
    """
    temp = temperature
    if filter_at_temperature:
        temp = filter_at_temperature

    entry_set = GibbsEntrySet.from_computed_entries(
        entries=entries,
        temperature=temp,
        include_nist_data=include_nist_data,
        include_freed_data=include_freed_data,
        include_icsd_entropy=include_icsd_entropy,
        ignore_nist_solids=ignore_nist_solids,
    )
    included_entries = [initialize_entry(f, entry_set) for f in formulas_to_include] if formulas_to_include else []

    entry_set = entry_set.filter_by_stability(e_above_hull=e_above_hull, include_polymorphs=include_polymorphs)
    entry_set.update(included_entries)  # make sure these aren't filtered out

    if filter_at_temperature and (filter_at_temperature != temperature):
        entry_set = entry_set.get_entries_with_new_temperature(temperature)

    if calculate_e_above_hulls:
        entry_set = GibbsEntrySet(deepcopy(entry_set), calculate_e_above_hulls=True)

    return entry_set


def initialize_entry(formula: str, entry_set: GibbsEntrySet, stabilize: bool = False):
    """Acquire an entry by user-specified formula. This method attemps to first
    get the entry; if it is not included in the set, it will create an interpolated
    entry. Finally, if stabilize=True, the energy will be lowered until it appears on
    teh hull.

    Args:
        formula: Chemical formula
        entry_set: Set of entries
        stabilize: Whether or not to stabilize the entry by decreasing its energy
            such that it is 'on the hull'.
    """
    try:
        entry = entry_set.get_min_entry_by_formula(formula)
    except KeyError:
        entry = entry_set.get_interpolated_entry(formula)
        warnings.warn(f"Using interpolated entry for {entry.composition.reduced_formula}")

    if stabilize:
        entry = entry_set.get_stabilized_entry(entry)

    return entry


def get_entries_from_entry_db(
    db: MongoStore,
    chemsys_formula_id_criteria: str | dict,
):
    """
    Warning:
        This function is meant for interacting with custom MongoDBs containing
        MP-compatible entries and is not broadly useful or applicable to other
        databases.

    Get a list of entries corresponding to a
    chemical system, formula, or materials_id or full criteria.

    Args:
        db: MongoStore object with database connection
        chemsys_formula_id_criteria: A chemical system
            (e.g., Li-Fe-O), or formula (e.g., Fe2O3) or materials_id (e.g., mp-1234) or
            full Mongo-style dict criteria.
    """
    if not isinstance(chemsys_formula_id_criteria, dict):
        criteria = parse_criteria(chemsys_formula_id_criteria)
    else:
        criteria = chemsys_formula_id_criteria

    entries = []
    for d in db.query(criteria):
        ents = d["entries"]
        if not ents:
            continue

        if ents.get("GGA+U"):
            e = ComputedStructureEntry.from_dict(ents["GGA+U"])
        elif ents.get("GGA"):
            e = ComputedStructureEntry.from_dict(ents["GGA"])
        else:
            logger.warning(f"Missing entry for {d['_id']}")
            continue
        entries.append(e)

    return entries


def get_all_entries_in_chemsys_from_entry_db(
    db: MongoStore,
    elements: str | list[str],
):
    """
    Warning:
        This function is meant for interacting with custom MongoDBs containing
        MP-compatible entries and is not broadly useful or applicable to other
        databases.

    Helper method for getting all entries in a total chemical system by querying
    database for all sub-chemical systems. Code adadpted from pymatgen.ext.matproj and
    modified to support very large chemical systems.

    Args:
        db: MongoStore object with database connection
        elements (str or [str]): Chemical system string comprising element
            symbols separated by dashes, e.g., "Li-Fe-O" or List of element symbols,
            e.g., ["Li", "Fe", "O"].
    """

    def divide_chunks(my_list, n):
        for i in range(0, len(my_list), n):
            yield my_list[i : i + n]

    if isinstance(elements, str):
        elements = elements.split("-")

    if len(elements) <= 13:
        all_chemsyses = []
        for i in range(len(elements)):
            for els in itertools.combinations(elements, i + 1):
                all_chemsyses.append("-".join(sorted(els)))

        all_chemsyses = list(divide_chunks(all_chemsyses, 1000))

        entries = []
        for chemsys_group in all_chemsyses:
            entries.extend(
                get_entries_from_entry_db(
                    db,
                    {"chemsys": {"$in": chemsys_group}},
                )
            )
    else:  # for very large chemical systems, use a different approach
        entries = get_entries_from_entry_db(
            db,
            {"elements": {"$not": {"$elemMatch": {"$nin": elements}}}},
        )

    return entries


def get_entries(
    db: MongoStore,
    chemsys_formula_id_criteria: str | dict,
    compatible_only: bool = True,
    inc_structure: str | None = None,
    property_data: list[str] | None = None,
    use_premade_entries: bool = False,
    conventional_unit_cell: bool = False,
    sort_by_e_above_hull: bool = False,
):  # pragma: no cover
    """Warning:
        This function is legacy code directly adapted from pymatgen.ext.matproj. It is
        not broadly useful or applicable to other databases. It is only used in jobs
        interfaced directly with internal databases at the Materials Project. This code
        is not adequately tested and may not work as expected.

    Get a list of ComputedEntries or ComputedStructureEntries corresponding to a
    chemical system, formula, or materials_id or full criteria.

    Args:
        db: MongoStore object with database connection
        chemsys_formula_id_criteria: A chemical system
            (e.g., Li-Fe-O), or formula (e.g., Fe2O3) or materials_id (e.g., mp-1234) or
            full Mongo-style dict criteria.
        compatible_only: Whether to return only "compatible"
            entries. Compatible entries are entries that have been processed using the
            MaterialsProjectCompatibility class, which performs adjustments to allow
            mixing of GGA and GGA+U calculations for more accurate phase diagrams and
            reaction energies.
        inc_structure: If None, entries returned are
            ComputedEntries. If inc_structure="initial", ComputedStructureEntries with
            initial structures are returned. Otherwise, ComputedStructureEntries with
            final structures are returned.
        property_data: Specify additional properties to include in
            entry.data. If None, no data. Should be a subset of supported_properties.
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
    props = ["final_energy", "unit_cell_formula", "task_id", *params]
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
                f"{d['pseudo_potential']['functional']} {label}" for label in d["pseudo_potential"].get("labels", [])
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
                prim = Structure.from_dict(d["initial_structure"] if inc_structure == "initial" else d["structure"])
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
            warnings.filterwarnings("ignore", message="Failed to guess oxidation states.*")
            entries = MaterialsProject2020Compatibility().process_entries(entries, clean=True)

    if sort_by_e_above_hull:
        entries = sorted(entries, key=lambda entry: entry.data["e_above_hull"])

    return entries


def get_all_entries_in_chemsys(
    db: MongoStore,
    elements: str | list[str],
    compatible_only: bool = True,
    inc_structure: str | None = "final",
    property_data: list | None = None,
    use_premade_entries: bool = False,
    conventional_unit_cell: bool = False,
    n: int = 1000,
) -> list[ComputedEntry]:  # pragma: no cover
    """Warning:
        This function is legacy code directly adapted from pymatgen.ext.matproj. It is
        not broadly useful or applicable to other databases. It is only used in jobs
        interfaced directly with internal databases at the Materials Project. This code
        is not adequately tested and may not work as expected.

    Helper method for getting all entries in a total chemical system by querying
    database for all sub-chemical systems. Code adadpted from pymatgen.ext.matproj and
    modified to support very large chemical systems.

    Args:
        db: MongoStore object with database connection
        elements (str or [str]): Chemical system string comprising element
            symbols separated by dashes, e.g., "Li-Fe-O" or List of element symbols,
            e.g., ["Li", "Fe", "O"].
        compatible_only (bool): Whether to return only "compatible"
            entries. Compatible entries are entries that have been processed using the
            MaterialsProjectCompatibility class, which performs adjustments to allow
            mixing of GGA and GGA+U calculations for more accurate phase diagrams and
            reaction energies.
        inc_structure (str): If None, entries returned are
            ComputedEntries. If inc_structure="final", ComputedStructureEntries with
            final structures are returned. Otherwise, ComputedStructureEntries with
            initial structures are returned.
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

    if len(elements) <= 13:
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
    """Parses a powerful and simple string criteria and generates a proper
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
