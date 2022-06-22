from pymatgen.core.composition import Composition as PymatgenComposition
from pymatgen.core.composition import reduce_formula
from functools import lru_cache


class Composition(PymatgenComposition):
    """Customized Composition class adapted from pymatgen"""

    def __init__(self, *args, strict: bool = False, **kwargs):
        super().__init__(*args, strict=strict, **kwargs)

    @lru_cache  # type: ignore
    def get_reduced_formula_and_factor(
        self, iupac_ordering: bool = False
    ) -> tuple[str, float]:
        """
        Calculates a reduced formula and factor.

        Args:
            iupac_ordering (bool, optional): Whether to order the
                formula by the iupac "electronegativity" series, defined in
                Table VI of "Nomenclature of Inorganic Chemistry (IUPAC
                Recommendations 2005)". This ordering effectively follows
                the groups and rows of the periodic table, except the
                Lanthanides, Actinides and hydrogen. Note that polyanions
                will still be determined based on the true electronegativity of
                the elements.

        Returns:
            A pretty normalized formula and a multiplicative factor, i.e.,
            Li4Fe4P4O16 returns (LiFePO4, 4).
        """
        all_int = all(
            abs(x - round(x)) < Composition.amount_tolerance for x in self.values()
        )
        if not all_int:
            return self.formula.replace(" ", ""), 1
        d = {k: int(round(v)) for k, v in self.get_el_amt_dict().items()}
        (formula, factor) = reduce_formula(d, iupac_ordering=iupac_ordering)

        if formula in Composition.special_formulas:
            formula = Composition.special_formulas[formula]
            factor /= 2

        return formula, factor
