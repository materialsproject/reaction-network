# ![Reaction Network](docs/images/logo.png)

Reaction network (rxn-network) is a Python package for predicting chemical reaction
pathways in solid-state materials synthesis using graph theory.

# Installing rxn-network

The rxn-network package has several dependencies, most of which can be installed
through PyPI. However, graph-tool must be installed through a more customized method;
please see https://graph-tool.skewed.de/ for more details. We recommend the
following installation procedure to create a new conda environment:

    conda create -n gt python=3.9

And then install graph-tool through conda-forge:

    conda install -c conda-forge graph-tool

To install an editable version of the rxn-network code, simply download (clone) the
code from this repository, navigate to its directory in terminal, and then run the
following command to install the requirements:

    pip install -r requirements.txt

And then to finally install an editable version of the package:

    pip install -e .

# Demo
A demo Jupyter notebook (demo.ipynb) contains the code necessary to replicate the
results of the paper and is a good starting template for using the rxn-network package.

**Note: the demo NB is currently being updated with refactored code; please see one of
the previous releases if you wish to use it!**


# How to cite rxn-network

The following paper explains the methodology of the rxn-network package:

*McDermott, M. J., Dwaraknath, S. S., and Persson, K. A. (2021). A graph-based network for predicting chemical reaction pathways in solid-state materials synthesis. Nature Communications, 12(1). https://doi.org/10.1038/s41467-021-23339-x*



# Acknowledgements

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of
Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: https://www.stonybrook.edu/genesis/
