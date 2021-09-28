#!/usr/bin/env python


from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="reaction-network",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Reaction-network is a Python package for predicting likely inorganic "
    "chemical reaction pathways using graph theory.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/GENESIS-EFRC/reaction-network",
    author="Matthew McDermott",
    author_email="mcdermott@lbl.gov",
    maintainer="Matthew McDermott",
    license="modified BSD",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"rxn_network": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "setuptools",
        "fireworks>=1.9.7",
        "maggma>=0.26.0",
        "numba>= 0.52.0",
        "pymatgen>=2022.0.10",
    ],
    extra_requires={"demo": ["jupyter>=1.0.0"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tests_require=["pytest"],
    python_requires=">=3.8",
)
