"""
Implementation of Fireworks for performing reaction enumreation and network
construction
"""
from typing import Iterable, Union, List, Dict, Optional
from fireworks import Firework

from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core.composition import Composition

from rxn_network.core import CostFunction, Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.firetasks.build_inputs import EntriesFromDb, EntriesFromMPRester
from rxn_network.firetasks.parse_outputs import ReactionsToDb, NetworkToDb
from rxn_network.firetasks.run_calc import (
    RunEnumerators,
    BuildNetwork,
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
        entry_set_params: Optional[Dict] = None,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """

        Args:
            enumerators:
            entries:
            chemsys:
            entry_set_params:
            db_file:
            entry_db_file:
            parents:
        """

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            tasks.append(
                _get_entry_task(
                    chemsys=chemsys,
                    entry_set_params=entry_set_params,
                    entry_db_file=entry_db_file,
                )
            )

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
        open_elem: Optional[str] = None,
        chempot: Optional[float] = None,
        solve_balanced_paths: bool = True,
        entry_set_params: Optional[Dict] = None,
        pathway_params: Optional[Dict] = None,
        solver_params: Optional[Dict] = None,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """

        Args:
            enumerators:
            cost_function:
            entries:
            chemsys:
            entry_set_params
            db_file:
            entry_db_file:
            include_polymorphs:
            parents:
        """
        pathway_params = pathway_params if pathway_params else {}
        solver_params = solver_params if solver_params else {}

        precursors = pathway_params.get("precursors")
        targets = pathway_params.get("targets")
        k = pathway_params.get("k")

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            tasks.append(
                _get_entry_task(
                    chemsys=chemsys,
                    entry_set_params=entry_set_params,
                    entry_db_file=entry_db_file,
                )
            )

        tasks.append(
            BuildNetwork(
                entries=entry_set,
                enumerators=enumerators,
                cost_function=cost_function,
                precursors=precursors,
                targets=targets,
                k=k,
                open_elem=open_elem,
                chempot=chempot,
            )
        )

        if solve_balanced_paths:
            try:
                net_rxn = Reaction(
                    [Composition(r) for r in precursors],
                    [Composition(p) for p in targets],
                )
            except ReactionError:
                raise ValueError(
                    "Can not balance pathways with specified precursors/targets."
                    "Please make sure a balanced net reaction can be written!"
                )

            solver = RunSolver(
                pathways=None,
                entries=entry_set,
                cost_function=cost_function,
                net_rxn=net_rxn,
                **solver_params,
            )
            tasks.append(RunSolver(solver))

        tasks.append(NetworkToDb(db_file=db_file))

        fw_name = f"Reaction Network (Targets: {targets}): {chemsys}"

        super().__init__(tasks, parents=parents, name=fw_name)


def _get_entry_task(chemsys, entry_set_params, entry_db_file):
    entry_set_params = entry_set_params if entry_set_params else {}
    temperature = entry_set_params.get("temperature", 300)
    e_above_hull = entry_set_params.get("e_above_hull", 0.0)
    include_polymorphs = entry_set_params.get("include_polymorphs", False)

    if bool(entry_db_file):
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

    return entry_task
