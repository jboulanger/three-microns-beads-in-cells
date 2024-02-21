# Detection of three microns beads in cells imaged in confocal microscopy

Cells labelled with DAPI and a membrane marker are segmented using cellpose. 
Large 3 microns beads are detected using normalized cross correlation. 

## Content
This repository contains a python source code and several notebooks
- beadfinder.py : code implementing bead detection and cell segmentation using cellpose
- Bead detections.py: jupyter notebook for processing files in a folder
- Bead detections hpc.py: jupyter notebook for processing files in a folder on a HPC system


## System requirement

Dependencies can be installed for example as:
```bash
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# reload the shell
${SHELL}
# create an environment
micromamba -qy create -f environment.yml
# activate the environment 
micromamba activate puroanalysis
# start the notebook
jupyter lab "Bead detections.csv"
```

To process data faster access to a HPC cluster can accelerate the processing.

## Installation

Download the [code and example data](https://github.com/jboulanger/three-microns-beads-in-cells/archive/refs/heads/main.zip) or clone the repository:
```bash
git clone https://github.com/jboulanger/three-microns-beads-in-cells.git
```

## Demo

