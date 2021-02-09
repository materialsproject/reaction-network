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
    description="Solid state reaction networks",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/GENESIS-EFRC/reaction-network",
    author="Matt McDermott",
    license="modified BSD",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"rxn_network": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "setuptools",
        "monty>=1.0.2",
        "numba>=0.50.1"
        "numpy>=1.18.4"
        "pymatgen>=2020.4.29"
        "scipy>=1.4.1"
        "tqdm>=4.46.0",
    ],
    extra_requires={"demo": ["jupyter>=1.0.0"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
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
    python_requires=">=3.6",
)
