from setuptools import find_packages, setup

develop = ["pre-commit", "sphinx-mdinclude"]
hydra_launchers = ["hydra-joblib-launcher", "hydra-submitit-launcher"]
zeus = ["zeus-mcmc", "h5py"]
all_extras = develop + hydra_launchers + zeus

setup(
    name="benchflow",
    version="0.0.1",
    description="Benchmarks for Sample-Efficient Bayesian Inference",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "scikit-learn",
        "lightning",
        "einops",
        "corner",
        "cma",
    ]
    + all_extras,
)
