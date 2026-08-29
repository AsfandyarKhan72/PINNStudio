from setuptools import setup, find_packages

setup(
    name="pinnstudio",
    version="1.1.0",
    author="AsfandyarKhan72",
    description="No-code GUI for Physics-Informed Neural Networks using DeepXDE",
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
