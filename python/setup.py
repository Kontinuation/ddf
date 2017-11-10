from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(Extension(
        'ddf',
        sources=['ddf.pyx'],
        language="c++",
        include_dirs = [numpy.get_include()],
        extra_compile_args=["-std=c++11"]))
)
