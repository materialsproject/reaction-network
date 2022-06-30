"""
Implementation of Fireworks for performing reaction enumreation and network
construction
"""
from typing import Dict, Iterable, Optional, Union

from fireworks import Firework

from rxn_network.core.composition import Composition
from rxn_network.core.cost_function import CostFunction
from rxn_network.core.enumerator import Enumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicOpenEnumerator
from rxn_network.enumerators.minimize import (
    MinimizeGibbsEnumerator,
    MinimizeGrandPotentialEnumerator,
)
from rxn_network.firetasks.build_inputs import get_entry_task
from rxn_network.firetasks.parse_outputs import NetworkToDb, ReactionsToDb
from rxn_network.firetasks.run_calc import (
    RunEnumerators,
    CalculateChempotDistance,
    CalculateSelectivity,
    BuildNetwork,
    RunSolver,
)
from rxn_network.reactions.basic import BasicReaction


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
        calculate_selectivity: bool = True,
        selectivity_kwargs: Optional[Dict] = None,
        calculate_chempot_distance: bool = False,
        db_file: str = ">>db_file<<",
        entry_db_file: str = ">>entry_db_file<<",
        parents=None,
    ):
        """
        Args:
            enumerators: List of enumerator objects to run
            entries: GibbsEntrySet object containing entries to use for enumeration
            chemsys: If entries aren't provided, they will be retrivied either from
                MPRester or from the entry_db corresponding to this chemical system.
            entry_set_params: Parameters to pass to the GibbsEntrySet constructor
            calculate_selectivity: Whether to calculate selectivity for the reactions
            selectivity_kwargs: Optional kwargs to pass to the CalculateSelectivity task
            calculate_chempot_distance: Whether to calculate the chemical potential
                distance as an additional selectivity metric. Defaults to False.
            db_file: Path to the database file to store the reactions in.
            entry_db_file: Path to the database file containing entries (see chemsys
                parameter above)
            parents: Parents of this Firework.
        """

        precursors = {
            precursor
            for enumerator in enumerators
            for precursor in enumerator.precursors
        }
        targets = {
            target for enumerator in enumerators for target in enumerator.targets
        }

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

            entry_set_params = entry_set_params or {}
            entry_set_params["formulas_to_include"] = list(precursors | targets)

            tasks.append(
                get_entry_task(
                    chemsys=chemsys,
                    entry_set_params=entry_set_params,
                    entry_db_file=entry_db_file,
                )
            )

        fw_name = (
            f"Reaction Enumeration (Precursors: {precursors}, Targets: {targets}):"
            f" {chemsys}"
        )

        tasks.append(
            RunEnumerators(
                enumerators=enumerators, entries=entry_set, task_label=fw_name
            )
        )

        if calculate_selectivity:
            selectivity_kwargs = selectivity_kwargs or {}

            for enumerator in enumerators:
                class_name = enumerator.__class__.__name__
                if class_name == "BasicOpenEnumerator":
                    if not selectivity_kwargs.get("open_phases"):
                        selectivity_kwargs["open_phases"] = enumerator.open_phases
                elif class_name == "MinimizeGibbsEnumerator":
                    if not selectivity_kwargs.get("use_minimize"):
                        selectivity_kwargs["use_minimize"] = True
                elif class_name == "MinimizeGrandPotentialEnumerator":
                    if not selectivity_kwargs.get("use_minimize"):
                        selectivity_kwargs["use_minimize"] = True
                    if not selectivity_kwargs.get("open_elem"):
                        selectivity_kwargs["open_elem"] = enumerator.open_elem
                        selectivity_kwargs["chempot"] = enumerator.mu

            tasks.append(CalculateSelectivity(entries=entries, **selectivity_kwargs))

        if calculate_chempot_distance:
            tasks.append(CalculateChempotDistance(entries=entries))

        tasks.append(ReactionsToDb(db_file=db_file, use_gridfs=True))

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
            net_rxn = BasicReaction.balance(
                [Composition(r) for r in precursors],
                [Composition(p) for p in targets],
            )
            if not net_rxn.balanced:
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
