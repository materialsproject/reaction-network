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
    Firework for running a list of enumerators and storing the resulting reactions in a database.
    An option is included to calculate competitiveness scores and store those as data
    within the reactions.
    """

    def __init__(
        self,
        enumerators: Iterable[Enumerator],
        entries: Optional[GibbsEntrySet] = None,
        chemsys: Optional[Union[str, Iterable[str]]] = None,
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
            entries: EntrySet object containing entries to use for enumeration
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
            if not chemsys:
                raise ValueError(
                    "If entries are not provided, a chemsys must be provided!"
                )
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

        tasks.append(RunEnumerators(enumerators=enumerators, entries=entry_set))

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

        fw_name = f"Reaction Enumeration (Targets: {targets}): {chemsys}"
        super().__init__(tasks, parents=parents, name=fw_name)


class NetworkFW(Firework):
    """
    Firework for building a ReactionNetwork and optionally performing pathfinding.
    Output data is stored within a database.
    """

    def __init__(
        self,
        enumerators: Iterable[Enumerator],
        cost_function: CostFunction,
        entries: Optional[GibbsEntrySet] = None,
        chemsys: Optional[Union[str, Iterable[str]]] = None,
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
            enumerators: List of enumerators to use for calculating reactions in the network.
            cost_function: A cost function used to assign edge weights to reactions in
                the network.
            entries: EntrySet object containing entries to use for enumeration (optional)
            chemsys: If entries aren't provided, they will be retrivied either from
                MPRester or from the entry_db corresponding to this chemsys.
            entry_set_params: Parameters to pass to the GibbsEntrySet constructor
            open_elem: Optional open element to use for renormalizing reaction energies.
            chempot: Optional chemical potential assigned to open_elem.
            solve_balanced_paths: Whether to solve for BalancedPathway objects using the
                PathwaySolver. Defaults to True.
            entry_set_params: Parameters to pass to the GibbsEntrySet constructor
            pathway_params: Parameters to pass to the ReactionNetwork.find_pathways() method.
            solver_params: Parameters to pass to the PathwaySolver constructor.
            db_file: Path to the database file to store the reaction network and pathways in.
            entry_db_file: Path to the database file containing entries (see chemsys argument)
            parents: Parents of this Firework.
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
            if not chemsys:
                raise ValueError(
                    "If entries are not provided, a chemsys must be provided!"
                )
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
            except ReactionError as e:
                raise ValueError(
                    "Can not balance pathways with specified precursors/targets."
                    "Please make sure a balanced net reaction can be written!"
                ) from e

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
