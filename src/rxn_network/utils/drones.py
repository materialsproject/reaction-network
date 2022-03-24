"""Drones for parsing firework data."""
from pymatgen.apps.borg.hive import AbstractDrone

from rxn_network.utils.models import EnumeratorTask, NetworkTask


class EnumeratorDrone(AbstractDrone):
    """
    Drone for parsing enumerated reaction data from an enumerator workflow into a task object.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def assimilate(self, path):
        """
        Assimilate the data and return the task.
        """
        d = EnumeratorTask.from_files(path / "rxnx.json.gz", path / "metadata.json.gz")
        return d

    def get_valid_paths(self, path):
        pass


class NetworkDrone(AbstractDrone):
    """
    Drone for parsing network data from a network workflow into a task object.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def assimilate(self, path):
        """
        Assimilate the data and return the task.
        """
        d = NetworkTask.from_files(path / "rxnx.json.gz", path / "metadata.json.gz")
        return d

    def get_valid_paths(self, path):
        pass
