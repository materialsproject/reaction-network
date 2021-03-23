from fireworks import Firework
from rxn_network.firetasks.build_inputs import EntriesFromDb, EntriesFromMPRester
from rxn_network.firetasks.run_calc import RunEnumerators
from rxn_network.firetasks.parse_outputs import ReactionsToDb
from rxn_network.entries.entry_set import GibbsEntrySet


class EnumeratorFW(Firework):
    """
    Firework for running a list of enumerators (which outputs a list of reactions).
    """
    def __init__(self, enumerators, entries_or_spec=None, db_file=None,
                 entry_db_file=None, parents=None):

        tasks = []

        if type(entries_or_spec) == dict:
            chemsys = entries_or_spec.get("chemsys")
            e_above_hull = entries_or_spec.get("e_above_hull")
            temperature = entries_or_spec.get("temperature")
            include_polymorphs = entries_or_spec.get("include_polymorphs", False)

            if entry_db_file:
                entry_task = EntriesFromDb(chemsys=chemsys,
                                           temperature=temperature,
                                           e_above_hull=e_above_hull,
                                           include_polymorphs=include_polymorphs)
            else:
                entry_task = EntriesFromMPRester(chemsys=chemsys,
                                                 temperature=temperature,
                                                 e_above_hull=e_above_hull,
                                                 include_polymorphs=include_polymorphs)
            tasks.append(entry_task)
        else:
            entry_set = GibbsEntrySet(entries)

        targets = [enumerator.target for enumerator in enumerators]

        chemsys = "-".join(sorted(list(entry_set.chemsys)))
        fw_name = f"Reaction Enumeration (Target: {targets}): {chemsys}"

        tasks.append(RunEnumerators(enumerators=enumerators, entries=entry_set,
                                    chemsys=chemsys))
        tasks.append(ReactionsToDb(db_file=db_file, calc_dir="."))

        super().__init__(tasks, parents=parents, name=fw_name)

class NetworkFW(Firework):
    pass