from setuptools import setup, Extension
import pybind11
import sys
import os

if sys.platform.startswith("win"):
    extra_compile_args = ['/openmp', '/O2', '/fp:fast']
    extra_link_args = []
else:
    extra_compile_args = ['-fopenmp', '-O3', '-ffast-math']
    extra_link_args = ['-fopenmp']

# Указываем полный путь к модулю через точки
ext_modules = [
    Extension(
        "__init__", 
        sources=["__init__.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    ext_modules=ext_modules,
)