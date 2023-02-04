# darkside50_BN

This repository contains the general C++ implementation for the analysis carried out in the paper "Search for low mass dark matter in DarkSide-50: the bayesian network approach".

To run this code the following inputs are needed:
* Theoretical energy spectra for background and signal components
* Observed data spectrum
* Asimov dataset
* The M2 matrices for the different backrgound and signal components
* Other detector-related informations (exposure, background activities, systematic corrections, etc.)

The output, in the form of the MCMC chains and other plots (posterior p.d.f.s, prior p.d.f.s, correlation plots, ...), will be saved in a new folder in the current directory.

Dependencies:
* BAT
* cnpy
* gsl
