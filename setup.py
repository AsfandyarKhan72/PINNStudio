from setuptools import setup, find_packages
from pathlib import Path
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="pinnstudio",
    version="1.5.1",
    author="AsfandyarKhan72",
    description="A no-code scientific computing environment for solving forward and inverse problems with physics-informed neural networks — from setting up the equations to training, error analysis, and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AsfandyarKhan72/PINNStudio",
    project_urls={"Source": "https://github.com/AsfandyarKhan72/PINNStudio", "Issues": "https://github.com/AsfandyarKhan72/PINNStudio/issues"},
    license="MIT",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent", "Intended Audience :: Science/Research", "Topic :: Scientific/Engineering :: Physics"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"pinnstudio": ["resources/*"]},
    install_requires=[
        "deepxde>=1.10.0",
        "torch>=2.0",
        "PyQt6>=6.4",
        # Newer PyQt6-Qt6 builds fail to load on some Windows versions with a
        # "DLL load failed" ImportError. Pin to a known-working build on Windows only;
        # Linux/macOS keep using the latest compatible version.
        "PyQt6==6.6.1; sys_platform == 'win32'",
        "PyQt6-Qt6==6.6.1; sys_platform == 'win32'",
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
    ],
    entry_points={
        "console_scripts": [
            "pinnstudio=pinnstudio.main:main",
        ],
    },
    python_requires=">=3.9",
)
