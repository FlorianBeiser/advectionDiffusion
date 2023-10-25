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

Utility functions are collected in the `.py`-files, but plots are generated in the `.ipynb`-notebooks.

### Forward model and truth generation
The truth is handled as an ensemble of size 1, the corresponding experimental set-up and the observed values are stored to file in `/experiment_files/...`. The timestamp serves as a key for each set-up or observation, respectively. See `Truth.ipynb`.

### Filtering 
Given a set-up (by its timestamp), data assimilation can be seen in action in `KFiltering.ipynb`, `ETKFiltering.ipynb`, and `SLETKFiltering.ipynb` for the respective methods. 

### Comparison
The `FilteringComparison.ipynb` opposes the results of the different data assimilation methods for a single experiment by different criteria.
The same critieria are evaulated for multiple repetitions in `run_FilteringComparison.py` with changing parameters if specified.


## Reproducing resutls 

The plots shown in the paper "Comparison of Ensemble-Based Data Assimilation Methods for Oceanographic Applications with Sparse Observations" are generated from this repository. 

- Figure 3.1: `Truth.ipynb` (Note that no seed is selected, such that the truth will look different every single realisation) and the values are written to file with a certain time stamp in the `/experiment_files/...`-folder. 
- Figure 3.2: `KFiltering.ipynb` (The truth is refered to by a time stamp.)
- Figure 3.3, 3.4, 3.7: `FilteringComparison.ipynb` (The truth is refered to by a time stamp and results are again stochastic.)
- Figure 3.5: `FilteringCoverage.ipynb`
- Figure 3.6: `Spectrum.ipynb`
- Figure 3.8, 3.9: Generate outputs with `run_FilteringComparisonLocalisation.py` and postprocess with `Localisation_PostProcessing.ipynb`
- Figure 3.10: Generate outputs with `run_FilteringComparison.py`

NB: The plots in the notebooks are only for reference and the final figures in the manuscript are generated with tikz such that they differ in style. 
