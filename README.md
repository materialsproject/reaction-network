# ![Reaction Network](docs/_static/img/logo.png)

![Codecov](https://img.shields.io/codecov/c/github/materialsproject/reaction-network?style=for-the-badge)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/materialsproject/reaction-network/testing.yml?style=for-the-badge)

![PyPI - Python
Version](https://img.shields.io/pypi/pyversions/reaction-network?style=for-the-badge)
![PyPI - Downloads](https://img.shields.io/pypi/dm/reaction-network?style=for-the-badge)
![PyPI - License](https://img.shields.io/pypi/l/reaction-network?style=for-the-badge)

Reaction Network (`rxn_network`) is a Python package for synthesis planning and predicting chemical reaction pathways in inorganic materials synthesis.

## Installation

We recommend installing using pip:

```properties
pip install -U reaction-network
```

The package will then be installed under the name `rxn_network`. The Materials Project
API is not installed by default; to install it, run: `pip install -U mp-api`.

> **Note**
> As of version 7.0 and beyond, the `reaction-network` package no longer uses `graph-tool`. All network functionality is now implemented using `rustworkx`. This means it is no longer required to complete any extra installations.

## Tutorials

The `examples` folder contains two (2) demonstration notebooks:

- **1_enumerators.ipynb**: how to enumerate reactions from a set of entries; running
  enumerators using jobflow
- **2_networks.ipynb**: how to build reaction networks from a list of enumerators and
  entries; how to perform pathfinding to recommend balanced reaction pathways; running
  reaction network analysis using jobflow

## Citation

If you use this code in your work, please consider citing the following paper (see
`CITATION.bib`):

> McDermott, M. J., Dwaraknath, S. S., and Persson, K. A. (2021). A graph-based network
> for predicting chemical reaction pathways in solid-state materials synthesis. Nature
> Communications, 12(1). <https://doi.org/10.1038/s41467-021-23339-x>

## Acknowledgements

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of
Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: <https://www.stonybrook.edu/genesis/>
