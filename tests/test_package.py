"""Package level: import, version, exception and warning hierarchy."""
import importlib
import inspect
import re

import pytest

import cytovanni
from cytovanni import exceptions


def test_version_is_semantic_string():
    assert isinstance(cytovanni.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+([.\-+].*)?", cytovanni.__version__)


def test_version_module_matches_package():
    from cytovanni.version import __version__
    assert __version__ == cytovanni.__version__


def test_version_source_is_version_py():
    """pyproject.toml takes the distribution version from version.py (hatch dynamic version)."""
    import os
    pyproject = os.path.join(os.path.dirname(cytovanni.__file__), "..", "..", "pyproject.toml")
    if not os.path.exists(pyproject):
        pytest.skip("not a source checkout")
    text = open(pyproject).read()
    assert re.search(r'dynamic\s*=\s*\[[^\]]*"version"', text)
    assert 'path = "src/cytovanni/version.py"' in text


@pytest.mark.parametrize("sub", ["exceptions", "utils", "torch", "io", "gating", "rainbow", "spill", "ref"])
def test_subpackages_are_attributes(sub):
    assert getattr(cytovanni, sub) is importlib.import_module(f"cytovanni.{sub}")


def _members(base):
    return [obj for _, obj in inspect.getmembers(exceptions, inspect.isclass)
            if obj.__module__ == exceptions.__name__ and obj is not base and issubclass(obj, base)]


def test_all_exceptions_derive_from_base():
    """Every exception class of the module is a CytovanniException (and an Exception), every warning a CytovanniWarning."""
    classes = [obj for _, obj in inspect.getmembers(exceptions, inspect.isclass) if obj.__module__ == exceptions.__name__]
    assert len(classes) > 10
    for cls in classes:
        if cls in (exceptions.CytovanniException, exceptions.CytovanniWarning):
            continue
        assert issubclass(cls, exceptions.CytovanniException) != issubclass(cls, exceptions.CytovanniWarning), cls
    assert issubclass(exceptions.CytovanniException, Exception)
    assert not issubclass(exceptions.CytovanniException, Warning)
    assert issubclass(exceptions.CytovanniWarning, Warning)


def test_exceptions_have_docstrings():
    for cls in _members(exceptions.CytovanniException) + _members(exceptions.CytovanniWarning):
        assert cls.__doc__ and cls.__doc__.strip(), cls


def test_exceptions_catchable_as_group():
    with pytest.raises(exceptions.CytovanniException):
        raise exceptions.ChannelMismatchException("x")


def test_warnings_filterable_as_group():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", exceptions.CytovanniWarning)
        warnings.warn("x", exceptions.GMMAssignmentWarning)
        warnings.warn("y", exceptions.RainbowSpilloverWarning)
    assert w == []


def test_no_duplicate_class_definitions_in_exceptions():
    """Each class name is defined once in exceptions.py."""
    src = inspect.getsource(exceptions)
    names = re.findall(r"^class (\w+)\(", src, re.M)
    assert sorted({n for n in names if names.count(n) > 1}) == []
