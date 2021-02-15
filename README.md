# Saccade-suppression-python

## Ibrahim Tunc Lab rotation project 'A basic model of image shift detection across species'
Aim of this project is to generate a model, which predicts species specific sensory noise in the eye position estimate. The predicted sensory noise is then aimed to be used to generate species-specific predictions in saccadic suppression using the Crevecoeur model.

### Structure
In the following, the content of each of the subfolder is explained:

### /data
This folder contains the figures (PDF) and model simulation files, along with the macaque superior colliculus anatomical data from Chen et al. 2019.

### /development_scripts
The scripts in this folder are used for test purposes and/or developing the main scripts, and can be completely ignored.

### /modelling
The scripts in this folder are used to generate and test the shift decoder models of different variants. There are two model variants. gabor_population.py includes the model where the visual field is regularly tiled into patches, and each patch contains the same set of filters with different size, spatial frequency and phase. The second model variant can be found in population_activity_shift_decoder.py, where the receptive field position is randomized for each Gabor-like unit, and each unit has random parameters chosen from a specific parameter distribution. population_activity_image_reconstruction is still under development.

### /saccade_detector
zf_saccade_analysis.py is used to detect and extract saccades for further analysis, and generates the figures located in /data. zf_plot_saccades.py can be used to check each saccade one by one.

### /simulations
scripts in this folder runs specific model simulations and saves them in the respective subfolder in \data, and generates violin plots based on the simulation results (scripts with the name ..._violinplot.py)

### zf_helper_funcs.py
This script contains all of the functions and classes used in all other scripts. Each function is documented in detail.

Each script contains comments, briefly explaining what each line is meant for. For questions and inquiries, please contact ibrahimalperentunc@protonmail.com

