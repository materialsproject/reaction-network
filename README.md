# [Reaction Network](docs/images/logo.png)

Reaction network (rxn-network) is a Python package for predicting chemical reaction pathways in solid-state materials synthesis using graph networks.

# Installing rxn-network

The rxn-network package has several software dependencies (listed in *requirements.txt*), most of which can be installed in less than a minute through PyPI. Note that graph-tool must be installed through a more customized method; please see https://graph-tool.skewed.de/ for more details, particularly if using a Windows based machine. 

Hence we recommend the following installation procedure which first creates a new conda environment:

    conda create -n gt python=3.8

And then installs graph-tool through conda-forge:

    conda install -c conda-forge graph-tool

Then simply download (clone) the reaction-network code from this repository, navigate to its directory in terminal, and then run the following commands to install the requirements and an (editable) version of the package:

    pip install -r requirements.txt
    pip install -e .

# Demo
A demo Jupyter notebook (demo.ipynb) contains the instructions necessary to replicate the results of the paper and is a good starting template for using the rxn-network package on your own systems.

# How to cite rxn-network

The following paper explains the methodology of the rxn-network package.
It is currently under review, but a preprint is accessible via:

https://assets.researchsquare.com/files/rs-38000/v1_stamped.pdf

# Acknowledgements

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of
Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: https://www.stonybrook.edu/genesis/
