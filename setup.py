import re
from setuptools import setup, find_packages
from pathlib import Path
this_dir = Path(__file__).parent
# the version is set in one place, pinnstudio/__init__.py
VERSION = re.search(r'^__version__ = "([^"]+)"', (this_dir / "pinnstudio" / "__init__.py").read_text(encoding="utf-8"),
                    re.M).group(1)
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="pinnstudio",
    version=VERSION,
    author="Asfandyar Khan",
    description="A graphical platform for forward and inverse physics-informed neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AsfandyarKhan72/PINNStudio",
    project_urls={"Source": "https://github.com/AsfandyarKhan72/PINNStudio", "Issues": "https://github.com/AsfandyarKhan72/PINNStudio/issues"},
    license_expression="MIT",             # SPDX license expression (PEP 639)
    license_files=["LICENSE", "THIRD_PARTY_LICENSES.md", "LICENSES/*"],
    classifiers=["Programming Language :: Python :: 3", "Programming Language :: Python :: 3.11", "Programming Language :: Python :: 3.12", "Programming Language :: Python :: 3.13", "Programming Language :: Python :: 3.14", "Operating System :: OS Independent", "Intended Audience :: Science/Research", "Topic :: Scientific/Engineering :: Physics"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"pinnstudio": ["resources/*"]},
    install_requires=[
        "deepxde>=1.10.0",
        "torch>=2.0",
        # PySide6-Essentials has every Qt module PINNStudio uses (QtCore, QtGui,
        # QtWidgets, QtSvg; QtTest for the tests) and none of the Addons. No
        # Windows pin is needed (the PyQt6 one was for "DLL load failed"):
        # 6.11 and 6.12 install and run on the Windows CI runners.
        # PySide6 6.12.0 loses a reference to None on every QTimer.singleShot
        # and to True on every signal emit(); harmless from Python 3.12 on
        # (immortal objects), fatal on 3.11 ("deallocating None").
        # tests/test_binding_refcounts.py checks the installed binding.
        "PySide6-Essentials>=6.5,<6.12; python_version < '3.12'",
        "PySide6-Essentials>=6.5; python_version >= '3.12'",
        # DeepXDE 1.15's 2D Polygon/Triangle geometry calls np.cross on
        # 2-element vectors, which NumPy deprecated in 2.0 and removed in
        # 2.5 (it now raises ValueError) -- keep NumPy below 2.5 until a
        # DeepXDE release fixes that.
        "numpy<2.5",
        # PyTorch's last Intel-Mac release (2.2.2) was built against NumPy 1
        # and cannot use NumPy 2 ("Failed to initialize NumPy").
        "numpy<2; sys_platform == 'darwin' and platform_machine == 'x86_64'",
        "matplotlib",
        "pandas",
        # imported directly (scipy.interpolate, scipy.spatial, scipy.special) by the GUI and the generated
        # scripts; previously installed only as a DeepXDE dependency
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "pinnstudio=pinnstudio.main:main",
        ],
    },
    python_requires=">=3.11",
)
