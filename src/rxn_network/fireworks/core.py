"""
Implementation of Fireworks for performing reaction enumreation and network
construction
"""
from typing import Dict, Iterable, Optional, Union

from fireworks import Firework
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core.composition import Composition

from rxn_network.core import CostFunction, Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.firetasks.build_inputs import get_entry_task
from rxn_network.firetasks.parse_outputs import NetworkToDb, ReactionsToDb
from rxn_network.firetasks.run_calc import (
    BuildNetwork,
    RunEnumerators,
    RunSolver,
    CalculateCScores,
)


class EnumeratorFW(Firework):
    """
    Firework for running a list of enumerators (which outputs a list of reactions).
    Option to calculate competitiveness scores and store those as data within the reactions.
    """

    def __init__(
        self,
        enumerators: Iterable[Enumerator],
        entries: GibbsEntrySet = None,
        chemsys: Union[str, Iterable[str]] = None,
        entry_set_params: Optional[Dict] = None,
        calculate_c_scores: Optional[Union[bool, int]] = False,
        cost_function: Optional[CostFunction] = None,
        c_score_kwargs: Optional[Dict] = None,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """

        Args:
            enumerators: List of enumerators to run
            entries: EntrySet to use for enumeration
            chemsys: If entries aren't provided, they will be retrivied either from
                MPRester or from the entry_db corresponding to this chemsys.
            entry_set_params: Parameters to pass to the GibbsEntrySet constructor
            calculate_c_scores: Whether or not to calculate c_scores; if an integer is
                provided, this will specify the maximum number of highest ranked reactions
                the calculation on.
            cost_function: The cost function used to rank reactions and then calculate c-scores.
            c_score_kwargs: Parameters to pass to the CompetitivenessScoreCalculator constructor.
            db_file: Path to the database file to store the reactions in.
            entry_db_file: Path to the database file containing entries (see chemsys
                parameter above)
            parents: Parents of this Firework.
        """

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            tasks.append(
                get_entry_task(
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

        if calculate_c_scores:
            if not cost_function:
                raise ValueError("Must provide a cost function to calculate C-scores!")

            c_score_kwargs = c_score_kwargs or {}

            for enumerator in enumerators:
                if isinstance(enumerator, BasicOpenEnumerator):
                    if not c_score_kwargs.get("open_phases"):
                        c_score_kwargs["open_phases"] = enumerator.open_phases
                elif isinstance(enumerator, MinimizeGibbsEnumerator):
                    if not c_score_kwargs.get("use_minimize"):
                        c_score_kwargs["use_minimize"] = True
                elif isinstance(enumerator, MinimizeGrandPotentialEnumerator):
                    if not c_score_kwargs.get("use_minimize"):
                        c_score_kwargs["use_minimize"] = True
                    if not c_score_kwargs.get("open_elem"):
                        c_score_kwargs["open_elem"] = enumerator.open_elem
                        c_score_kwargs["chempot"] = enumerator.mu

            c_score_kwargs.update(
                {
                    "entries": entry_set,
                    "cost_function": cost_function,
                    "k": calculate_c_scores,
                }
            )
            tasks.append(CalculateCScores(**c_score_kwargs))

        tasks.append(ReactionsToDb(db_file=db_file))

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
            open_elem:
            chempot:
            solve_balanced_paths:
            entry_set_params:
            pathway_params:
            solver_params:
            db_file:
            entry_db_file:
            parents:
        """
        pathway_params = pathway_params if pathway_params else {}
        solver_params = solver_params if solver_params else {}

        precursors = pathway_params.get("precursors", [])
        targets = pathway_params.get("targets", [])
        k = pathway_params.get("k")

        tasks = []

        entry_set = None
        if entries:
            entry_set = GibbsEntrySet(entries)
            chemsys = "-".join(sorted(list(entry_set.chemsys)))
        else:
            tasks.append(
                get_entry_task(
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
