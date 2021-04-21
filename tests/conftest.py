def pytest_itemcollected(item):
    """Make tests names more readable in the tests output."""
    item._nodeid = (
        item._nodeid.replace(".py", "")
        .replace("tests/", "")
        .replace("test_", "")
        .replace("_", " ")
        .replace("Test", "")
        .replace("Class", " class")
        .lower()
    )
    doc = item.obj.__doc__.strip() if item.obj.__doc__ else ""
    if doc:
        item._nodeid = item._nodeid.split("::")[0] + "::" + doc