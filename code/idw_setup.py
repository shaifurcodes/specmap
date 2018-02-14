from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'IDW',
  ext_modules = cythonize("idw.pyx"), include_dirs=[numpy.get_include()]
)
