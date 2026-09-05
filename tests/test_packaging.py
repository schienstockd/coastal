"""setuptools packaging catches subpackages that would silently miss the wheel.

The explicit `packages = ["coastal"]` in the original pyproject.toml shipped a wheel WITHOUT
`coastal.support` when it moved in from cecelia (#30) — pytest passed against the source tree but
downstream `pip install` saw the subpackage vanish, and cecelia's runner blew up with
`ModuleNotFoundError: No module named 'coastal.support'`. Auto-discovery fixes it; this test pins
the fix so the next subpackage cannot regress it silently.
"""
from setuptools import find_packages


def test_discovery_covers_every_subpackage():
    """Every dir under coastal/ that has an __init__.py must ship in the wheel."""
    pkgs = set(find_packages(where='.', include=['coastal', 'coastal.*']))
    assert 'coastal' in pkgs
    assert 'coastal.support' in pkgs, \
        f'coastal.support missing from find_packages output — packaging regressed (got {sorted(pkgs)})'
