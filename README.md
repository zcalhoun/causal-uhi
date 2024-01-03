# Causal UHI

This repository contains the code used to perform experiments for the manuscript *Estimating the effects of vegetation and increased
albedo on the urban heat island effect with spatial causal inference* [todo -- add link to this once published].

In this repository, there are:
1. Three scripts that demonstrate the [cross-validation strategy](calc_optimal_weights.py), [bootstrapping method](bootstrap_coefficients.py), and [joint training algorithm](joint_train.py) used to obtain final results.
2. Several [notebooks](notebooks) that were used to analyze results and obtain final visualizations.
3. [Dataset loaders](Datasets/durham.py) to demonstrate pre-processing steps used in our analysis.
4. TIFF files taken from the original data sources and formatted to be in the same coordinate reference system, so that the data can be used together. These files can be found in the [data/durham](data/durham) folder.
5. Bash scripts that were used to run experiments (found in the [scripts](scripts) directory). You should be able to download this repository and execute these scripts to replicate my results.

# Questions?
Please create an issue if you have a question about the code, and I will try my best to get back to you.
