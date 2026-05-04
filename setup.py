from setuptools import setup, find_packages

setup(
    name="pinnstudio",
    version="0.1.0",
    author="AsfandyarKhan72",
    description="No-code GUI for Physics-Informed Neural Networks using DeepXDE",
    packages=find_packages(),
    install_requires=[
        "deepxde>=1.10.0",
        "torch>=2.0",
        "PyQt6>=6.4",
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
