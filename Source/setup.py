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
        "BatSpec.Core.Spectral.Statistic.FastZScore", 
        sources=[os.path.join("BatSpec", "Core", "Spectral", "Statistic", "FastZScore.cpp")],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    name="BatSpec",
    packages=["BatSpec"], # Говорим setuptools, что это пакет
    ext_modules=ext_modules,
)