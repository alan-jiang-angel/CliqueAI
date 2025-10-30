# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="bk_core",
    ext_modules=cythonize("bk_core.pyx", compiler_directives={'language_level': "3"}),
)