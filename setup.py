"""Minimal setup.py.

The metadata lives in pyproject.toml. This file exists only to mark the
distribution as *non-pure* (it contains the pre-compiled extension _axon.pyd),
so that the generated wheel gets the correct platform tag.

Building the extension itself is done by scripts/build_python.ps1 (MinGW), and the
resulting _axon.pyd is included as package-data (see pyproject.toml).
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # noqa: D401
        return True


setup(distclass=BinaryDistribution)
