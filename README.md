# ![Reaction Network](docs/images/logo.png)

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/GENESIS-EFRC/reaction-network/testing?style=for-the-badge)
![Codecov](https://img.shields.io/codecov/c/github/GENESIS-EFRC/reaction-network?style=for-the-badge)

![PyPI - Python
Version](https://img.shields.io/pypi/pyversions/reaction-network?style=for-the-badge)
![PyPI - Downloads](https://img.shields.io/pypi/dm/reaction-network?style=for-the-badge)
![PyPI - License](https://img.shields.io/pypi/l/reaction-network?style=for-the-badge)

Reaction Network (`rxn_network`) is a Python package for predicting likely inorganic chemical reaction pathways using graph theoretical methods.

# Installation directions

This package can be easily installed using pip:

```properties
pip install reaction-network
```

The package will then be installed under the name `rxn_network`. 

### Warning :warning:
While this will take care of most dependencies, if you are using any of the network-based features, then the `graph-tool` package must be installed. Unfortunately, this cannotbe installed through pip. Please see <https://graph-tool.skewed.de/> for more details. :warning:

We recommend the following installation procedure which installs graph-tool through conda-forge.

```properties
conda install -c conda-forge graph-tool
```

# Tutorial notebooks

The `examples` folder contains two (2) demonstration notebooks:

- **1_enumerators.ipynb**: how to enumerate reactions from a set of entries; running
  enumerators using jobflow
- **2_network.ipynb**: how to build reaction networks from a list of enumerators and
  entries; how to perform pathfinding to recommend balanced reaction pathways; running
  reaction network analysis using jobflow

# Citation

If you use this code or Python package in your work, please consider citing the following paper:

> McDermott, M. J., Dwaraknath, S. S., and Persson, K. A. (2021). A graph-based network for predicting chemical reaction pathways in solid-state materials synthesis. Nature Communications, 12(1). <https://doi.org/10.1038/s41467-021-23339-x>

# Acknowledgements

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of
Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: <https://www.stonybrook.edu/genesis/>
