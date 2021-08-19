# advectionDiffusion
Propagation of a Gauss-linear model and data assimilation by means of the analytical Kalman filter (KF) as well as by means of the ensemble-based Ensemble transform Kalman filter (ETKF) without and with localisation 

## Acknowledgement
The example and the original code is taken from the work of Gunhild Berget (NTNU). 

## Content
The stochastic advection diffusion equations serves as an example to compare the posteriors of the ETKF and the LETKF, respectively, with the analytical posteriors of the KF.

## Requirements
We recommend the use of a conda environment with Python 3.8 (tested) and the following packages
- jupyter 
- numpy
- matplotlib
- scipy
- statsmodels

## Introduction to code structure

### Forward model and truth generation
The truth is handled as an ensemble of size 1, the corresponding experimental set-up and the observed values are stored to file in `/experiment_files/...`. The timestamp serves as a key for each set-up or observation, respectively. See `Truth.ipynb`.

### Filtering 
Given a set-up (by its timestamp), data assimilation can be seen in action in `KFiltering.ipynb`, `ETKFiltering.ipynb`, and `SLETKFiltering.ipynb` for the respective methods. 

### Comparison
The `FilteringComparison.ipynb` opposes the results of the different data assimilation methods by different criteria.
