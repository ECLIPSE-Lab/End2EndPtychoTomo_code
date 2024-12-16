# ✨ Code for:Near-Isotropic Sub-Ångstrom 3D Resolution Phase Contrast Imaging Achieved by End-to-End Ptychographic Electron Tomography  ✨
[![Paper](https://img.shields.io/badge/Phys_Scr_(2024)-b31b1b.svg)](https://iopscience.iop.org/article/10.1088/1402-4896/ad9a1a)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


📄 <u> Read our paper here: </u>\
[**Near-Isotropic Sub-Ångstrom 3D Resolution Phase Contrast Imaging Achieved by End-to-End Ptychographic Electron Tomography**](https://iopscience.iop.org/article/10.1088/1402-4896/ad9a1a)\
*Shengbo You, Andrey Romanov, Philipp Pelz*

## Requirements

- taichi
- pytorch
- abtem
- ase
- h5py
- matplotlib
- numpy
- scipy
- pyprismatic
- kornia
- zarr
- [ccpi](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit) (ccpi uses [Apache 2.0 license](https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit?tab=Apache-2.0-1-ov-file#readme))

## Install

``` bash
pip install . -e
```
## Data links

- Raw data: [link](https://zenodo.org/records/13060513)
- Alignment results: [link](https://zenodo.org/records/14499409)
- MSP Reconstructed results: [link](https://zenodo.org/records/14499409)

## Overview

### Simulation 
build_PtAl2O3_xyzFile.py: build the xyz file of the PtAl2O3 nanoparticle

build_PtAl2O3_volume.py: build the volume of the PtAl2O3 nanoparticle

PtAl2O3_DataCollection.py: simulate 4DSTEM data collection of the PtAl2O3 nanoparticle

general_interface_ptychotomo_PtAl2O3.py: reconstruct the PtAl2O3 nanoparticle using the joint ptychotomo model

### Experiment

Alignment_Te.py: align the tilt series of the Te nanoparticle

general_interface_ptychotomo_Te.py: reconstruct the Te nanoparticle using the joint ptychotomo model