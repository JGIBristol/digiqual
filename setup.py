import shutil
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class CustomBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            compiler_type = self.compiler.compiler_type
            if compiler_type == "msvc":
                ext.extra_compile_args.extend(["/O2", "/std:c++17", "/D_USE_MATH_DEFINES"])
            else:
                ext.extra_compile_args.extend(["-O3", "-std=c++17"])

        super().build_extensions()


class CustomBuildPy(build_py):
    """Bundles the Shiny GUI (the top-level app/ directory) into the
    installed package as digiqual/app/, so `dq_ui()` can find it at
    runtime. app/ lives outside src/ and is its own separate uv project
    (desktop-build + local-dev tooling), so find_packages(where="src")
    never sees it -- this replicates what the pre-C++ hatchling build did
    via `[tool.hatch.build.targets.wheel.force-include]`, which has no
    setuptools equivalent.
    """

    def run(self):
        super().run()
        dest = Path(self.build_lib) / "digiqual" / "app"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2("app/app.py", dest / "app.py")
        shutil.copy2("app/run_app.py", dest / "run_app.py")
        shutil.copytree("app/www", dest / "www", dirs_exist_ok=True)


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
    version="0.26.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt, "build_py": CustomBuildPy},
)
