from fireworks import Firework
from rxn_network.firetasks.run_calc import RunEnumerators
from rxn_network.firetasks.parse_outputs import ReactionsToDb
from rxn_network.entries.entry_set import GibbsEntrySet


class EnumeratorFW(Firework):
    """
    Firework for running a list of enumerators (which outputs a list of reactions).
    """
    def __init__(self, enumerators, entries, db_file=None, parents=None):
        entry_set = GibbsEntrySet(entries)
        targets = [enumerator.target for enumerator in enumerators]

        chemsys = "-".join(sorted(list(entry_set.chemsys)))
        fw_name = f"Reaction Enumeration (Target: {targets}): {chemsys}"

        tasks = []
        tasks.append(RunEnumerators(enumerators=enumerators, entries=entry_set))
        tasks.append(ReactionsToDb(db_file=db_file, calc_dir="."))
        super().__init__(tasks, parents=parents, name=fw_name)

class NetworkFW(Firework):
    pass