"""Tests for pathway plotter"""
import pytest

from rxn_network.pathways.plotting import PathwayPlotter


@pytest.fixture()
def phase_amts():
    return {
        "Y2O3": [0.5, 0.5, 0.495, 0.485, 0.45, 0.375, 0.25, 0.05, 0.025, 0.0],
        "Mn2O3": [0.5, 0.5, 0.495, 0.485, 0.45, 0.375, 0.25, 0.05, 0.025, 0.0],
        "YMnO3": [0.0, 0.0, 0.01, 0.03, 0.10, 0.25, 0.5, 0.90, 0.95, 1.0],
    }


@pytest.fixture()
def temps():
    return [800 + 20 * i for i in range(10)]


@pytest.fixture()
def plotter(phase_amts, temps):
    """Create plotter"""
    return PathwayPlotter(phase_amts, temps)


def test_plot_pathway(plotter):
    plotter.plot_pathway()


def test_plot_energy_cascade(plotter, mp_entries):
    plotter.plot_energy_cascade(mp_entries)
