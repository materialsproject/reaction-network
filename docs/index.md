<img alt="Reaction Network" src="images/logo.png" width="600">

Reaction network (rxn-network) is a Python package for predicting chemical reaction pathways in solid-state materials synthesis using graph networks.

## Manuscript & Results
This version/release of the rxn-network package was used in the accompanying manuscript, which is under review at Nature Communciations:

McDermott, M.J., Dwaraknath, S.S., and Persson, K.A. (2021). A graph-based network for predicting chemical reaction pathways in solid-state materials synthesis.

This version of the repository includes results from the paper in the _results_ folder.

##### Note: This code is in active development, and edits to the the code will be available here as new releases in the future.

## Installing rxn-network

The rxn-network package has several software dependencies (see the *requirements.txt* file), most of which can be installed in less than a minute through PyPI. Note that graph-tool must be installed through a more customized method; please see https://graph-tool.skewed.de/ for more details, particularly if using a Windows based machine. 

We recommend the following installation procedure for those using an OSX or Linux based machine. First create a new conda environment (here named *gt*) and activate it:

    conda create -n gt python=3.8
    conda activate gt

And then install graph-tool through conda-forge (this may take 1-2 minutes):

    conda install -c conda-forge graph-tool

Then simply download (clone) the reaction-network code from this repository, navigate to its directory and run the following commands to install the requirements and an (editable) version of the package:

    pip install -r requirements.txt
    pip install -e .

## Demo
A demo Jupyter notebook (demo.ipynb) contains the instructions necessary to replicate the results of the manuscript and is a good starting template for using the rxn-network package on your own systems. Simply start a Jupyter notebook server and launch the notebook file.

## Contact
For questions concerning this project, please either raise an Issue on the Github repository or contact the maintainer via email: mcdermott *[at]* lbl.gov.

## Acknowledgement

This work was supported as part of GENESIS: A Next Generation Synthesis Center, an 
Energy Frontier Research Center funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award Number DE-SC0019212.

Learn more about the GENESIS EFRC here: https://www.stonybrook.edu/genesis/
