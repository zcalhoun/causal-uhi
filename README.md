# Causal UHI

This repository contains the code used to perform experiments for the manuscript [Estimating the effects of vegetation and increased
albedo on the urban heat island effect with spatial causal inference](https://www.nature.com/articles/s41598-023-50981-w).

In this repository, there are:
1. Three scripts that demonstrate the [cross-validation strategy](calc_optimal_weights.py), [bootstrapping method](bootstrap_coefficients.py), and [joint training algorithm](joint_train.py) used to obtain final results.
2. Several [notebooks](notebooks) that were used to analyze results and obtain final visualizations.
3. [Dataset loaders](Datasets/durham.py) to demonstrate pre-processing steps used in our analysis.
4. TIFF files taken from the original data sources and formatted to be in the same coordinate reference system, so that the data can be used together. These files can be found in the [data/durham](data/durham) folder.
5. Bash scripts that were used to run experiments (found in the [scripts](scripts) directory). You should be able to download this repository and execute these scripts to replicate my results.

# Getting Started
To run this code, you need to create a coding environment with the dependencies listed below, and then you should create a bash script to run experiments based on the desired experiment arguments.

## Dependencies
The requirements for running the code contained in this repository are relatively small -- you can run the main 3 python scripts for replication results with just the first three packages below:
* rasterio - for loading data
* scipy - for certain signal processing operations
* sklearn - for ridge regression, gaussian processes, k-means, etc.
* cartopy - for creating maps (only required for certain notebooks)
* geopandas - alternative method for loading data (this is only required in some of the notebooks)

## Running experiments
Please refer to the following scripts to replicate the results in the publication.
* [Joint training of model](joint_train.py) -- this script iterates between ridge regression fitting and Gaussian process fitting, so that you can review model convergence. See bash script [here](scripts/joint_train.sh).
* [Bootstrapping the coefficients](bootstrap_coefficients.py) -- this script will apply block k-means to the data to implement block bootstrapping. As a result, we obtain confidence intervals on the ridge regression coefficients.See bash script [here](scripts/bootstrap_final.sh), and a notebook analyzing the results [here](notebooks/bootstrap_figures.ipynb).
* [Cross validation](calc_optimal_weights.py) -- this script applies block cross validation to the desired hyperparameters, and saves the results of each run so that the optimal hyperparameters can be found. Refer to example bash scripts [here](scripts/final_cross_val_scripts/). Note that there are multiple scripts here, as a lot of hyperparameters were tested (very simply parallelized by breaking up hyperparameters into multiple scripts and executing on different virtual machines). See notebook analyzing results [here](notebooks/cv_results.ipynb). Lastly, see visualization of the block clusters used for cross validation and block bootstrapping in [this notebook](notebooks/cross_val_viz.ipynb).

Once you have a bash script created, you can simply run that bash script from the terminal and check the results directory to see the results.

# References
The data found in the [data/durham](data/durham) folder relies on the following sources:
1. Dewitz, J., 2023, National Land Cover Database (NLCD) 2021 Products: U.S. Geological Survey data release, https://doi.org/10.5066/P9JZ7AO3.
2. North Carolina State Climate Office, 2021. Urban Heat Island Temperature Mapping Campaign, https://climate.ncsu.edu/research/uhi/.
3. Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone.

# Questions?
Please create an issue if you have a question about the code, and I will try my best to get back to you.
