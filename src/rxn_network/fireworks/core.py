"""
Implementation of Fireworks for performing reaction enumreation and network
construction
"""
from fireworks import Firework

from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.firetasks.build_inputs import EntriesFromDb, EntriesFromMPRester
from rxn_network.firetasks.parse_outputs import ReactionsToDb

from rxn_network.firetasks.run_calc import RunEnumerators


class EnumeratorFW(Firework):
    """
    Firework for running a list of enumerators (which outputs a list of reactions).
    """

    def __init__(
        self,
        enumerators,
        entries=None,
        chemsys=None,
        temperature=None,
        e_above_hull=None,
        db_file=None,
        entry_db_file=None,
        include_polymorphs=False,
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

        targets = {enumerator.target for enumerator in enumerators}
        if len(targets) != 1:
            raise ValueError("Enumerators contain different targets!")
        target = targets.pop()
        fw_name = f"Reaction Enumeration (Target: {target}): {chemsys}"

        tasks.append(
            RunEnumerators(enumerators=enumerators, entries=entry_set, chemsys=chemsys)
        )
        tasks.append(ReactionsToDb(db_file=db_file, calc_dir="."))

        super().__init__(tasks, parents=parents, name=fw_name)


class NetworkFW(Firework):
    pass
