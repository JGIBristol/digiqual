import sys
import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

class CustomBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            compiler_type = self.compiler.compiler_type
            if compiler_type == 'msvc':
                ext.extra_compile_args.extend(['/O2', '/std:c++17'])
            else:
                ext.extra_compile_args.extend(['-O3', '-std=c++17'])

        super().build_extensions()

ext_modules = [
    Pybind11Extension(
        "digiqual._digiqual_cpp",
        [
            "src/cpp/bindings.cpp",
            "src/cpp/kernel_smoothing.cpp",
            "src/cpp/mc_integration.cpp",
        ],
        include_dirs=[
            "src/cpp",
        ],
        cxx_std=17,
    ),
]

setup(
    name="digiqual",
    version="0.24.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
