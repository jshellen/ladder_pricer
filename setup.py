from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

sources = [
    "cpp/src/python_bindings.cpp",
    "cpp/src/matrix.cpp",
    "cpp/src/core.cpp",
    "cpp/src/howard.cpp",
    "cpp/src/analytics.cpp",
    "cpp/src/simulation.cpp",
]

ext_modules = [
    Pybind11Extension(
        "trinity._native",
        sources,
        include_dirs=["cpp/include"],
        cxx_std=17,
        extra_compile_args=["-O3", "-Wall", "-Wextra", "-Wpedantic"],
    )
]

setup(
    name="trinity-ladder-pricer",
    version="0.1.0",
    description="FX ladder pricer: C++ Howard engine with Dash UI",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=["dash>=3.0", "plotly>=6.0", "numpy>=1.24"],
    extras_require={"test": ["pytest>=8"]},
)
