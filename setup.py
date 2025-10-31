from setuptools import setup, find_packages

setup(
    name="vasp-mace",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "vasp-mace=vasp_mace.cli:main",
        ],
    },
    install_requires=[
        "ase",
        "torch",
        "mace-torch",
    ],
)
