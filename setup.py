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
    # NOTE: `torch-extras` (ScatterNet/layers' former contents, now a
    # separate local package) is also a required runtime dependency, but
    # isn't published yet, so it can't be listed here as a normal
    # `install_requires` entry without hardcoding a local path. Install it
    # separately first: `pip install -e ../torch-extras` (also covered by
    # requirements.txt).
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "h5py",
        "hdf5plugin",
        "xraydb",
        "beartype",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "scatternet-train=Train.train:main",
        ],
    },
)
