# ![Reaction Network](docs/images/logo.png)

[![Pytest Status](https://github.com/GENESIS-EFRC/reaction-network/workflows/testing/badge.svg)](https://github.com/GENESIS-EFRC/reaction-network/actions?query=workflow%3Atesting)
[![Code Coverage](https://codecov.io/gh/GENESIS-EFRC/reaction-network/branch/main/graph/badge.svg)](https://codecov.io/gh/GENESIS-EFRC/reaction-network)

Reaction network (rxn-network) is a Python package for predicting chemical reaction
pathways in solid-state materials synthesis using combinatorial and graph-theorteical methods.

# Installation directions

The rxn-network package has several dependencies, most of which can be installed
through PyPI. However, graph-tool must be installed through a more customized method;
please see https://graph-tool.skewed.de/ for more details. We recommend the
following installation procedure which creates a new conda environment with Python 3.9,
activates it, and then installs graph-tool through conda-forge.

```properties
conda create -n gt python=3.9
conda activate gt
conda install -c conda-forge graph-tool
```

To install an editable version of the rxn-network code, simply download (clone) the
code from this repository, navigate to its directory, and then run the
following command to install the requirements:

```properties
pip install -r requirements.txt
pip install -e .
```

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
