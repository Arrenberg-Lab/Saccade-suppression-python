# Python code for estimating sensory and motor noise in zebrafish and macaque in saccadic behavior

Aim of this project is to estimate species-specific sensory and motor noise parameters during saccadic behavior according to the Crevecoeur model (Crevecoeur Kording 2017).

### Structure
In the following, the content of each of the subfolder is explained:

## /data
This folder contains the figures (PDF) and model simulation files, along with the macaque superior colliculus anatomical data from Chen et al. 2019 and macaque saccade data from Hafed Lab.

## /development_scripts
The scripts in this folder are used for test purposes and/or developing the main scripts, and can be completely ignored. Some of the important scripts are the following:

#### 1) GaussianRf.py 
Script used for developing 2D Gaussian RFs in geographical coordinates.

#### 2) DOG_generate_rfs_test.py 
Test script used for generating Difference of Gaussians (DoG) RFs.

#### 3) DOG__rf_new_settings_test.py 
Test script used for generating Difference of Gaussians (DoG) RFs.

#### 4) DoG_fourier_spectrum.py 
Test script used to generate the lookup table for DoG spatial frequency and standard deviation values of the Gaussians used in DoG. 

#### 5) sphere_test.py 
Test script for the geographical transformations to make sure Mercator plots we use are correct. Figures here are to check the correctness of the transformations (spherical, cartesian, cylindric).

#### 6) coordinate_trsf_before_after.py 
Comparison of the visual field images with RFs with and without geographical coordinate transformations. 1 figure showing the visual fields in both cases.

#### 7) Giulias_stimulus.py 
Development script for implementing the random pixel motion stimulus in the cylindrical area (stimulus used by Giulia Soto). The important aspects implemented in this script are transferred to zf_helper_funcs.py.

## /modelling
The scripts in this folder are used to generate and test the shift decoder models of different variants. The variations are RF profile (Gaussian or Gabor), visual field (flat or spherical), parameter distributions for the RFs as well as the model architecture (Rechardt-like or center-of-mass-decoder). The scripts in this folder are:
#### 1) popualtion_activity_shift_decoder.py
This is an example script, showing how the sensory shift detection model from the laboratory rotation (center of mass stimulus position decoder) works, using Gabor filters and NO geographic coordinate transformations.
#### FIGURES 
Visual field with all RFs, 5 example stimulus shifts with real and model stimulus position readouts along with a scatterplot of real and decoded stimulus shifts. The final figure shows the distribution of stimulus shift decoding errors pooled over all stimulus shift magnitudes.
#### 2) plot_zf_mac_rfs.py
This script generates an example model of RFs for zebrafish and macaque, using the settings from lab rotation (see also population_activity_shift_decoder.py). 
#### FIGURES 
One figure 2x2 showing for zebrafish and macaque the RFs in whole visual field (upper) and in a zoomed region (lower). One figure showing the stimulus shifts. One 2x2 figure showing the parameter histograms for zebrafish (upper) and macaque (lower). 


## /saccade_detector
zf_saccade_analysis.py is used to detect and extract saccades for further analysis, and generates the figures located in /data. zf_plot_saccades.py can be used to check each saccade one by one.

## /server_codes

## /simulations
scripts in this folder run specific model simulations and saves them in the respective subfolder in \data, as well as generate violin plots based on the simulation results (scripts with the name ..._violinplot.py)

## zf_helper_funcs.py and zf_helper_funcs_.py
These script contains all of the functions and classes used in all other scripts. Each function is documented in detail. zf_helper_funcs_.py is the same as zf_helper_funcs.py, with slight changes in some of the functions. This script was generated while developing the server codes.

Each script contains comments, briefly explaining what each line is meant for. For questions and inquiries, please contact ibrahimalperentunc@protonmail.com

### Ibrahim Tunc, 03.05.2022
