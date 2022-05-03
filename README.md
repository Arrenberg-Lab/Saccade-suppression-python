# Python code for estimating sensory and motor noise in zebrafish and macaque in saccadic behavior

Aim of this project is to estimate species-specific sensory and motor noise parameters during saccadic behavior according to the Crevecoeur model (Crevecoeur Kording 2017).

Each script contains comments, briefly explaining what each line is meant for. For questions and inquiries, please contact ibrahimalperentunc@protonmail.com

## Explanation of the dataset structures used externally in the project:

For the motor noise estimations, zebrafish and macaque saccade recordings were used. Zebrafish saccades were recorded by Giulia Soto, and saccade dataset was obtained from Hafed Lab. The brief explanation of the structure of these dataset can be found in the following:

#### 1) zebrafish dataset
The dataset is located in the server (\\server_ip_replace_value\arrenberg_data\shared\Ibrahim\HSC_saccades), with each recording of a given fish is in a folder with the name HSC_saccades_etc. In each of these folders, there are additional folders for each of the detected saccade for the given fish. In each of these saccade-specific folders, there are metadata files along with a txt file containing eye positions in degrees (eyepos_etc.txt), which is extracted for saccade detection.
Function extract_saccade_data from zf_helper_funcs automatically extracts all the eye traces in the server folder.

#### 2) macaque dataset
The macaque dataset is a .mat file and can be found in /data/Hafed_dataset/example_saccades.mat. The file contains a time array (mytime), and arrays (nsaccades x ntime) for horizontal as well as vertical components of the saccades in different directions (up/down + left/right). The preprocessing and loading of the dataset is achieved in the script macaque_saccade_detection.py. Metadata and readme for the dataset can further be found in the .mat file.

#### 3) simulation results
The raw data of the simulations (RF arrays, RF activity arrays, visual field etc.) are saved as numpy arrays (.npy/.npz). Processed data (model decoding error etc.) are saved as pandas dataframe files, with each entry having unique values about the simulation settings (animal type, number of RFs, stimulus shift magnitude etc.).

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

#### 3) gabor_population.py
The earliest version of the position decoder model, where the Gabor filters with different spatial frequency and phase are tested. The main idea is to reconstruct the stimulus based on the overlap of the Gabor filter activities. This script can be silently ignored. 

#### 4) model_geographic.py
The center of mass stimulus position decoder model in geographical coordinates (-> visual field is spherical).
#### FIGURES 
1 figure showing the decoding error of the model over multiple trials. 

#### 5) zf_test_coinc_det.py
Simplified version of the Reichardt-like motion decoder model for zebrafish. This script is used for debugging purposes. 
#### FIGURES
Figure with 2 subplots, one showing the coverage map, and the other one showing the motion decoder results. An additional figure shows the velocity decoding error histogram over 10 trials. 

#### 6) test_1D_rfs.py
Generate 1D RFs for zebrafish & macaque. This script was used in the first phase of developing zf_test_coinc_det.py. Therefore this script is rather obsolete.
#### FIGURES
1 figure for zebrafish with 2 subplots, one showing unit activites and other showing RF coverage along azimuth. 1 figure for macaque similar to that of zebrafish.

#### 7) test_1D_coinc_det.py
1D-reduced version of the zf_test_counc_det.py. In this version RFs are 1D Gaussians along azimuth. 
#### FIGURES
1 figure with 3 subplots, one showing RF activities, other showing visual field coverage map, and the last one showing the motion decoder results. Boolean parameter 'check' determines, if additional figures are generated, which show the stimulus in each frame (debugging purposes). 

#### 8) model_initial_improved.py
Example script for Reichardt-like motion decoder model (detailed explanation can be found under srv_run_simul_for_animals_given_seed.py). 
#### FIGURES
Example activity of the RF set for a given frame, parameter distributions for the RFs, net motion signal figure, figure showing the decoding error for all simulations.

## /saccade_detector
This folder includes the scripts used for processing zebrafish and macaque saccades as well as for estimating multiplicative and additive Crevecoeur motor noise variables.

#### 1) zf_saccade_detection_v2.py
Refined version of the zebrafish saccade detection (from Nyström & Holmqvist 2010). The dataset is recorded by Giulia Soto and can be found in server (\\server_ip_replace_value\arrenberg_data\shared\Ibrahim\HSC_saccades). 
#### FIGURES 
This script can generate plots for raw saccades & the onset/offset detections, plots showing for each saccade the additive & mutliplicative noises, as well as the plots showing saccades with a decent fit which are chosen for motor noise estimations. The boolean parameter 'plot' in the script determines which plots are skipped, by default it is False overall. Switching it to True before a specific figure line would activate the generation of that specific plot.

#### 2) macaque_saccade_detection.py
Very similar to zf_saccade_detection_v2.py, all motor noise-related plots are generated here. Each saccade can be inspected by changing the boolean parameter 'checksaccades'.
#### FIGURES 
Overshoot distribution of each saccadic direction (up/down + left/right), pooled overshoot distribution, multiplicative motor noise figure, additive motor noise distribution

## /server_codes
These scripts were developed to run Reichardt-like detector simulations in remote server, which was not carried out after finding out the detector model is not suitable for estimating sensory noise. The scripts are added just for the sake of possible recycling:

* srv_run_simul_for_animals_given_seed.py => Runs the sensory simulation for the given seed, animal and other parameter settings using Reichardt-like detector model (Frechette et al. 2004). This script first generates the RFs, then runs the simulations for each RF, and finally motion decoding is done. This is the master script to run simulations in the server. The RFs are Gaussian, the parameters are chosen from the updated distributions, and geographical transformation is implemented (i.e. visual field is shown in Mercator plot). Also each RF comes in phase-shifted pairs.

#### 1) srv_master_script.py
Reduced version of srv_run_simul_for_animals_given_seed.py (1 animal and 1 seed), for possible debugging purposes.

#### 2) srv_generate_RFs.py 
Generate and save each RF for the server simulations. This script is run within srv_run_animals_given_seed.py.

#### 3) generate_rfs_for_all.py 
Same as srv_generate_RFs.py, but not embedded within the server pipeline, such that it is easier in this version to manipulate parameters. This script can be mostly used for debugging purposes. Similarly, for given settings each RF is saved in separate files. 

#### 4) srv_simulate_rfs_with_stimulus.py 
Simulation of the RFs with the simple sweeping bar stimulus fpr the server simulations. This script is run within srv_run_animals_given_seed.py. Decoded results are then saved in separate files.

#### 5) srv_motion_decoder.py 
The implementation of the Reichardt-like motion detector. This script is run within srv_run_animals_given_seed.py. Decoding results are then saved in separate files.

#### 6) srv_check_visual_field_coverage.py
Check the coverage of the visual field along azimuth for the given parameter combinations. This script generates a binary coverage map figure, where if an azimuth angle is covered by at least 1 RF the coverage is 1 and otherwise 0.

#### 7) srv_violinplot_results.py
Plot the server simulation results as a violinplot. 
!!! I did not use this script yet, since we did not run extensive simulations in the server. Hence there might be some bugs.

## /simulations
scripts in this folder run specific model simulations and saves them in the respective subfolder in \data, as well as generate violin plots based on the simulation results (scripts with the ending ...violinplot.py):

#### 1) species_model_simulations_randomstimloc.py
Run center of mass decoder moder simulations, where the stimulus is randomly shifted along azimuth only.

#### 2) species_model_simulations_randomstimloc_plots.py 
This script generates __horizontal shift__ sensory noise __figures__ in my lab rotation report, based on the results from species_model_simulations_randomstimloc.py.

#### 3) species_model_simulations.py
This script runs the simulations for the center of mass stimulus position decoder. Here, the stimulus starts at visual field center, and is shifted along the same values of azimuth and elevation. The boolean script 'simulate' (which is True by default) determines if the simulatiins are run. By setting it to False, this script can be used to only generate the RF parameter distributions for zebrafish and macaque. The figure is nonetheless generated when simulate == True.
FIGURES : 1 Figure 2x2 subplots, showing the RF parameter distributions for zebrafish (upper) and macaque (lower). 

#### 4) species_model_violinplots.py
Generate the violinplots for the simulations from species_model_simulations.py. These violinplots were used for the __oblique shift__ sensory noise __figures__ in my lab rotation report.

#### 5) shift_decoder_simulations.py 
Runs simulations very similar to species_model_simulations.py but with different stimulus shift values.

#### 6) shift_decoder_violin_plot.py
Generates the violin plots based on the simulations from shift_decoder_simulations.py.

#### 7) species_model_violinplots.py
This script generates all the sensory noise related figures, where stimulus displacement is along a diagonal going through visual field center. The simulations here are based on Gabor RFs, WITHOUT geographic transformations, and based on the old parameter distribution, which was updated based on feedback & inputs from Ari.


## zf_helper_funcs.py and zf_helper_funcs_.py
These script contains all of the functions and classes used in all other scripts. Each function is documented in detail. zf_helper_funcs_.py is the same as zf_helper_funcs.py, with slight changes in some of the functions. This script was generated while developing the server codes.

## Open issues (as of 03.05.2022)
- The sensory noise is yet to be estimated using updated settings (new parameter distributions, spherical visual field, Gaussian RF profiles).

### Ibrahim Tunc, 03.05.2022
