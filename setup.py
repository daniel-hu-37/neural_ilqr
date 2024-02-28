#!/usr/bin/env python
"""iLQR setup."""

import os
from setuptools import setup, find_packages


def read(fname):
  """Reads a file's contents as a string.

    Args:
        fname: Filename.

    Returns:
        File's contents.
    """
  return open(os.path.join(os.path.dirname(__file__), fname)).read()


INSTALL_REQUIRES = [
    "numpy>=1.16.3", "scipy>=1.2.1", "jax>=0.2.17", "jaxlib>=0.1.71"
]

setup(name="neural_ilqr",
      version="1.0",
      description="Auto-differentiated Iterative Linear Quadratic Regulator",
      long_description=read("README.rst"),
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      install_requires=INSTALL_REQUIRES)
