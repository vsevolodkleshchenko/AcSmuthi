import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'cython_speedups',
        sources=['cython_speedups.pyx'],
    )
]

setup(
    name='cython_speedups',
    # ext_modules=cythonize(ext_modules, annotate=True)
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)

