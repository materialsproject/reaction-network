"""Test pydantic models for builders"""

import pytest

from rxn_network.builders.schema import (
    ComputedSynthesisRecipe,
    ComputedSynthesisRecipesDoc,
    Phase,
    Reaction,
    ReactionCompetition,
    ReactionHeuristics,
    ReactionThermo,
)
from rxn_network.core.composition import Composition


def test_phase_from_computed_entry(filtered_entries):
    """Test Phase.from_computed_entry"""
    phase = Phase.from_computed_entry(
        filtered_entries.get_min_entry_by_formula("YMnO3")
    )
    assert phase.formula == "YMnO3"
    assert not phase.is_metastable
    assert phase.temp == 1000


def test_reaction_from_computed_rxn(computed_rxn):
    """Test Reaction.from_computed_rxn"""
    rxn = Reaction.from_computed_rxn(computed_rxn)
    assert rxn.rxn_str == "2 YClO + 2 NaMnO2 + 0.5 O2 -> Y2Mn2O7 + 2 NaCl"
    assert len(rxn.reactants) == 3
    assert len(rxn.products) == 2


def test_computed_synthesis_recipe_from_computed_rxn(computed_rxn):
    """Test ComputedSynthesisRecipe.from_computed_rxn"""
    recipe = ComputedSynthesisRecipe.from_computed_rxn(
        computed_rxn, cost=0.1, target=Composition("Y2Mn2O7")
    )
    assert recipe.cost == pytest.approx(0.1)
    assert recipe.byproducts == ["NaCl"]
