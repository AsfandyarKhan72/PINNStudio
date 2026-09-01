from setuptools import setup, find_packages
from pathlib import Path
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="pinnstudio",
    version="1.1.0",
    author="AsfandyarKhan72",
    description="No-code GUI for Physics-Informed Neural Networks using DeepXDE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AsfandyarKhan72/PINNStudio",
    project_urls={"Source": "https://github.com/AsfandyarKhan72/PINNStudio", "Issues": "https://github.com/AsfandyarKhan72/PINNStudio/issues"},
    license="MIT",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent", "Intended Audience :: Science/Research", "Topic :: Scientific/Engineering :: Physics"],
    packages=find_packages(),
    install_requires=[
        "deepxde>=1.10.0",
        "torch>=2.0",
        "PyQt6>=6.4",
        # Newer PyQt6-Qt6 builds fail to load on some Windows versions with a
        # "DLL load failed" ImportError. Pin to a known-working build on Windows only;
        # Linux/macOS keep using the latest compatible version.
        "PyQt6==6.6.1; sys_platform == 'win32'",
        "PyQt6-Qt6==6.6.1; sys_platform == 'win32'",
        "numpy",
        "matplotlib",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "pinnstudio=pinnstudio.main:main",
        ],
    },
    python_requires=">=3.9",
)
