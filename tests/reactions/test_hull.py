""" Tests for InterfaceReactionHull. """
import numpy as np
import pytest
from pymatgen.core.composition import Element

from rxn_network.core.composition import Composition
from rxn_network.reactions.hull import InterfaceReactionHull


@pytest.fixture(scope="module")
def stable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 2 BaO -> Ba2TiO4":
            return r


@pytest.fixture(scope="module")
def unstable_rxn(bao_tio2_rxns):
    for r in bao_tio2_rxns:
        if str(r) == "TiO2 + 0.9 BaO -> 0.1 Ti10O11 + 0.9 BaO2":
            return r


def test_stable_reactions(irh_batio):
    stable_rxns = [
        "BaO -> BaO",
        "TiO2 + 2 BaO -> Ba2TiO4",
        "TiO2 + BaO -> BaTiO3",
        "TiO2 + 0.5 BaO -> 0.5 BaTi2O5",
        "TiO2 + 0.3077 BaO -> 0.07692 Ba4Ti13O30",
        "TiO2 + 0.2 BaO -> 0.2 BaTi5O11",
        "TiO2 + 0.1667 BaO -> 0.1667 BaTi6O13",
        "18 TiO2 -> O2 + 2 Ti9O17",
    ]

    stable_rxns = [
        actual_rxn
        for r in stable_rxns
        for actual_rxn in irh_batio.reactions
        if r == str(actual_rxn)
    ]
    assert irh_batio.stable_reactions == stable_rxns


def test_unstable_reactions(irh_batio):
    assert set(irh_batio.unstable_reactions) | set(irh_batio.stable_reactions) == set(
        irh_batio.reactions
    )


@pytest.mark.parametrize(
    "c1, c2, c3, expected_altitude",
    [
        ((0.0, 0.0), (0.5, -1.0), (1.0, 0.0), -1.0),
        ((0.25, -0.5), (0.4, -0.8), (0.9, -0.63), -0.27),
        ((0.33333, -0.23), (0.67777, 5.50), (0.8, 1.0), 4.822161),
    ],
)
def test_calculate_altitude(c1, c2, c3, expected_altitude):
    assert InterfaceReactionHull._calculate_altitude(c1, c2, c3) == pytest.approx(
        expected_altitude
    )


@pytest.mark.parametrize(
    "x1, x2, expected_length",
    [(0, 1, 9), (0.3, 1, 9), (0.8, 0.9, 4), (0.9, 1, 3), (0.3, 0.4, 2)],
)
def test_get_coords_in_range(x1, x2, expected_length, irh_batio):
    coords = irh_batio.get_coords_in_range(x1, x2)
    assert len(coords) == expected_length
    assert coords[0, 0] == pytest.approx(x1)
    assert coords[-1, 0] == pytest.approx(x2)


def test_get_primary_selectivity(irh_batio, stable_rxn, unstable_rxn):
    assert irh_batio.get_primary_selectivity(stable_rxn, temp=300) == pytest.approx(
        1.7824197647022084
    )
    assert irh_batio.get_primary_selectivity(unstable_rxn, temp=300) == pytest.approx(
        41.321218932490815
    )


def test_get_secondary_selectivity(irh_batio, stable_rxn, unstable_rxn):
    assert irh_batio.get_secondary_selectivity(stable_rxn) == pytest.approx(
        0.41705577943708827
    )
    assert irh_batio.get_secondary_selectivity(unstable_rxn) == pytest.approx(
        1.4664328079494886
    )


def test_get_energy_above_hull(irh_batio, stable_rxn, unstable_rxn):
    assert irh_batio.get_energy_above_hull(stable_rxn) == pytest.approx(0.0)
    assert irh_batio.get_energy_above_hull(unstable_rxn) == pytest.approx(
        1.0365808689708862
    )

    for r in irh_batio.unstable_reactions:
        assert (
            irh_batio.get_energy_above_hull(r) > -1e-12
        )  # some numerical error expected here

    for r in irh_batio.stable_reactions:
        assert irh_batio.get_energy_above_hull(r) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0, 1, -322.0267006467209),
        (0, 0.0001, 0.0),
        (0, 0.428, 0.0),
        (0, 0.430, -0.0011368030287497444),
        (0, 0.500, -0.0488825302218695),
        (0.5, 0.8, -0.06433079481994658),
    ],
)
def test_get_decomposition_energy(x1, x2, expected, irh_batio):
    assert irh_batio.get_decomposition_energy(x1, x2) == pytest.approx(expected)


def test_get_decomposition_energy_and_num_paths_recursive(irh_batio):
    (
        decomp_energy,
        num_paths,
    ) = irh_batio.get_decomposition_energy_and_num_paths_recursive(0, 1)

    assert decomp_energy == pytest.approx(irh_batio.get_decomposition_energy(0, 1))
    assert num_paths == pytest.approx(
        irh_batio.count(len(irh_batio.get_coords_in_range(0, 1)) - 2)
    )


@pytest.mark.parametrize(
    "num, answer",
    [(0, 1), (1, 1), (2, 2), (3, 5), (6, 132), (12, 208012), (18, 477638700)],
)
def test_count(num, answer, irh_batio):
    assert irh_batio.count(num) == irh_batio._count_recursive(num)[0] == answer


def test_hull_vertices(irh_batio):
    correct = np.array([1, 5, 19, 28, 43, 54, 57, 117])
    np.testing.assert_almost_equal(irh_batio.hull_vertices, correct)

    assert 89 not in irh_batio.hull_vertices  # 89 is above zero and not relevant


def test_plot(irh_batio):
    irh_batio.plot()
