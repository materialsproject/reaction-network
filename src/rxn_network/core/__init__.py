" Core interfaces for RXN Network"
from rxn_network.core.reaction import Reaction
from rxn_network.core.pathway import Pathway


class CostFunction(MSONable, metaclass=ABCMeta):
    " Base definition for a cost function "

    @abstractmethod
    def evaluate(self, rxn: Reaction) -> float:
        " Evaluates the cost function on a reaction "


class Enumerator(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction enumeration methodology "

    @abstractmethod
    def estimate_num_reactions(self, entries) -> int:
        " Estimate of the number of reactions from a list of entires "

    @abstractmethod
    def enumerate(self, entries) -> List[Reaction]:
        " Enumerates the potential reactions from the list of entries "



class ReactionNetwork(MSONable, metaclass=ABCMeta):
    " Base definition for a reaction network "

    def __init__(self, entries: List[Entry], enumerators, cost_function):

        self.entries = entries
        self.enumerators = enumerators
        self.cost_function = cost_function

    @abstractmethod
    def find_best_rxn_pathways(self, precursors, targets, num=15):
        " Find the N best reaction pathways "