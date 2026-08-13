"""Setuptools packaging config for the scatternet package.

Declares the `scatternet` distribution, its `ScatterNet`, `Preprocess`,
and `Train` sub-packages, runtime dependencies, and the
`scatternet-train` console-script entry point.
"""

from setuptools import find_packages, setup

setup(
    name="scatternet",
    version="0.1.0",
    packages=find_packages(
        include=[
            "ScatterNet",
            "ScatterNet.*",
            "Preprocess",
            "Preprocess.*",
            "Train",
            "Train.*",
        ]
    ),
    install_requires=[
        "torch",
        # ScatterNet/layers' former contents: https://pypi.org/project/pyteals/
        "pyteals",
        "numpy",
        "scipy",
        "h5py",
        "hdf5plugin",
        "xraydb",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "scatternet-train=Train.train:main",
        ],
    },
)
