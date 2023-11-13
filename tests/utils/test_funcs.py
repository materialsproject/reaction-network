"""Test for utils funcs"""


from rxn_network.utils.funcs import datetime_str, get_project_root


def test_get_project_root():
    assert get_project_root().name == "src"


def test_datetime_str():
    assert datetime_str()
