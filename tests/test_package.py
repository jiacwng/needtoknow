# The package imports and reports the version declared in pyproject.toml.

import tomllib
from pathlib import Path

import needtoknow


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert needtoknow.__version__ == declared
