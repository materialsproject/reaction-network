"""
Implementation of Fireworks for performing reaction enumreation and network
construction
"""
from typing import Iterable, Union, List, Dict
from fireworks import Firework

from rxn_network.core import CostFunction, Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.firetasks.build_inputs import EntriesFromDb, EntriesFromMPRester
from rxn_network.firetasks.parse_outputs import ReactionsToDb, NetworkToDb
from rxn_network.firetasks.run_calc import (
    RunEnumerators,
    BuildNetwork,
    FindPathways,
    RunSolver,
)


class EnumeratorFW(Firework):
    """
    Firework for running a list of enumerators (which outputs a list of reactions).
    """

    def __init__(
        self,
        enumerators: Iterable[Enumerator],
        entries: GibbsEntrySet = None,
        chemsys: Union[str, Iterable[str]] = None,
        entry_set_params: Dict = None,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """

        Args:
            enumerators:
            entries:
            chemsys:
            temperature:
            e_above_hull:
            db_file:
            entry_db_file:
            include_polymorphs:
            parents:
        """

        entry_set_params = entry_set_params if entry_set_params else {}
        temperature = entry_set_params.get("temperature", 300)
        e_above_hull = entry_set_params.get("e_above_hull", 0.0)
        include_polymorphs = entry_set_params.get("include_polymorphs", False)

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            if entry_db_file:
                entry_task = EntriesFromDb(
                    entry_db_file=entry_db_file,
                    chemsys=chemsys,
                    temperature=temperature,
                    e_above_hull=e_above_hull,
                    include_polymorphs=include_polymorphs,
                )
            else:
                entry_task = EntriesFromMPRester(
                    chemsys=chemsys,
                    temperature=temperature,
                    e_above_hull=e_above_hull,
                    include_polymorphs=include_polymorphs,
                )
            tasks.append(entry_task)

        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }

        fw_name = f"Reaction Enumeration (Targets: {targets}): {chemsys}"

        tasks.append(
            RunEnumerators(enumerators=enumerators, entries=entry_set, chemsys=chemsys)
        )
        tasks.append(ReactionsToDb(db_file=db_file, calc_dir="."))

        super().__init__(tasks, parents=parents, name=fw_name)


class NetworkFW(Firework):
    """
    Firework for building a reaction network and performing (optional) pathfinding.
    """

    def __init__(
        self,
        enumerators: Iterable[Enumerator],
        cost_function: CostFunction,
        entries: GibbsEntrySet = None,
        chemsys: Union[str, Iterable[str]] = None,
        entry_set_params=None,
        find_pathways: bool = True,
        solve_balanced_paths: bool = True,
        pathway_params=None,
        solver_params=None,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """

        Args:
            enumerators:
            entries:
            chemsys:
            temperature:
            e_above_hull:
            db_file:
            entry_db_file:
            include_polymorphs:
            parents:
        """

        entry_set_params = entry_set_params if entry_set_params else {}
        temperature = entry_set_params.get("temperature", 300)
        e_above_hull = entry_set_params.get("e_above_hull", 0.0)
        include_polymorphs = entry_set_params.get("include_polymorphs", False)

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            if entry_db_file:
                entry_task = EntriesFromDb(
                    entry_db_file=entry_db_file,
                    chemsys=chemsys,
                    temperature=temperature,
                    e_above_hull=e_above_hull,
                    include_polymorphs=include_polymorphs,
                )
            else:
                entry_task = EntriesFromMPRester(
                    chemsys=chemsys,
                    temperature=temperature,
                    e_above_hull=e_above_hull,
                    include_polymorphs=include_polymorphs,
                )
            tasks.append(entry_task)

        tasks.append(
            BuildNetwork(enumerators=enumerators, entries=entry_set, chemsys=chemsys)
        )
        tasks.append(NetworkToDb(db_file=db_file, calc_dir="."))

        if find_pathways:
            if not pathway_params:
                pathway_params = {}
            tasks.append(FindPathways())

        if solve_balanced_paths:
            if not solver_params:
                solver_params = {}
            tasks.append(RunSolver(solver))

        fw_name = f"Reaction Network (Target: {target}): {chemsys}"
        super().__init__(tasks, parents=parents, name=fw_name)
