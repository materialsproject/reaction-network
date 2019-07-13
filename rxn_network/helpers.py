import numpy as np

from pymatgen.analysis.reaction_calculator import ComputedReaction

from monty.json import MSONable


class RxnEntries(MSONable):

    def __init__(self, entries, description):
        self._entries = entries

        if description in ["r", "reactants", "Reactants"]:
            self._description = "Reactants"
        elif description in ["p", "products", "Products"]:
            self._description = "Products"
        elif description in ["s", "starters", "Starters"]:
            self._description = "Starters"
        elif description in ["t", "target", "Target"]:
            self._description = "Target"

    @property
    def entries(self):
        return self._entries

    @property
    def description(self):
        return self._description

    def __repr__(self):
        return f"{self._description}: {str([entry.composition.reduced_formula for entry in self._entries])}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash((self._description, frozenset(self._entries)))

class RxnPathway(MSONable):

    def __init__(self, rxns, weights):
        self._rxns = list(rxns)
        self._weights = list(weights)

        self.total_weight = sum(self._weights)
        self._dH_per_atom = [rxn.calculated_reaction_energy / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
                             for rxn in self._rxns]

    @property
    def rxns(self):
        return self._rxns

    @property
    def weights(self):
        return self._weights

    @property
    def dH_per_atom(self):
        return self._dH_per_atom

    def __repr__(self):
        path_info = ""
        for rxn, dH in zip(self._rxns, self._dH_per_atom):
            path_info += f"{rxn} (dH = {round(dH,3)} eV/atom) \n"

        path_info += f"Total Cost: {round(self.total_weight,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._rxns))


class CombinedPathway(MSONable):

    def __init__(self, paths, targets=None):
        self._paths = paths
        self.average_weight = np.mean([path.total_weight for path in self._paths])
        self.total_weight = sum([path.total_weight for path in self._paths])
        self._targets = targets
        self.net_rxn = self.get_net_rxn()

    def get_net_rxn(self):
        reactants = []
        products = []
        for path in self._paths:
            for step in path.rxns:
                reactants.extend(step._reactant_entries)
                products.extend(step._product_entries)

        if set(self._targets).issubset(set(products)):
            try:
                rxn = ComputedReaction(reactants, products)
            except:
                rxn = None
        else:
            rxn = None

        return rxn

    @property
    def paths(self):
        return self._paths

    def __repr__(self):
        path_info = ""
        for path in self._paths:
            path_info += f"{str(path)} \n\n"
        path_info += f"Average Cost: {round(self.average_weight,3)} \nTotal Cost: {round(self.total_weight,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._paths))