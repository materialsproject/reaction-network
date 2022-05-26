# ![Reaction Network](docs/images/logo.png)

[![Pytest Status](https://github.com/GENESIS-EFRC/reaction-network/workflows/testing/badge.svg)](https://github.com/GENESIS-EFRC/reaction-network/actions?query=workflow%3Atesting)
[![Code Coverage](https://codecov.io/gh/GENESIS-EFRC/reaction-network/branch/main/graph/badge.svg)](https://codecov.io/gh/GENESIS-EFRC/reaction-network)

Reaction network (rxn-network) is a Python package for predicting chemical reaction
pathways in solid-state materials synthesis using combinatorial and graph-theorteical methods.

# Installation directions

This package can be easily installed using pip:

```properties
pip install reaction-network
```

:warning: While this will take care of most dependencies, if you are using any of the network-based features (i.e. within `rxn_network.network`), then `graph-tool` must be installed. Unfortunately, this cannot
be installed through pip; please see https://graph-tool.skewed.de/ for more details. :warning:

We recommend the following installation procedure which installs graph-tool through conda-forge.

```properties
conda install -c conda-forge graph-tool
```

## For developers: 
To install an editable version of the rxn-network code, simply clone the
code from this repository, navigate to its directory, and then run the
following command to install the requirements:

```properties
pip install -r requirements.txt
pip install -e .
```

Note that this only works if the repository is cloned from GitHub, such that it contains
the proper metadata.

# Tutorial notebooks

The `notebooks` folder contains two (2) demonstration notebooks: 
- **enumerators.ipynb**: how to enumerate reactions from a set of entries; running
  enumerators using Fireworks
- **network.ipynb**: how to build reaction networks from a list of enumerators and
  entries; how to perform pathfinding to recommend balanced reaction pathways; running
  reaction network analysis using Fireworks

# Citation 

If you use this code or Python package in your work, please consider citing the following paper:

> McDermott, M. J., Dwaraknath, S. S., and Persson, K. A. (2021). A graph-based network for predicting chemical reaction pathways in solid-state materials synthesis. Nature Communications, 12(1). https://doi.org/10.1038/s41467-021-23339-x


# Acknowledgements

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of
Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: https://www.stonybrook.edu/genesis/
