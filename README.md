# Neuro-CNN: Convolutional Neural Networks for neural localization and classification

Neuro-CNN is a project for using machine learning, in particular Convolutional Neural Networks (CNN), to localize neurons in
3d and to classify their neural types from the Extracellular Action Potential (EAP) on Multi-Electrode Arrays (MEA).

To clone this repo open your terminal and run:

`git clone https://github.com/CINPLA/NeuroCNN.git`

## Pre-requisites

Neuro-CNN runs on Python2 and Python3. In order to install all required packages we recommend creating an anaconda
(https://www.anaconda.com/download/#linux) environment using the environment files. Open your terminal and run:

For Anaconda2
`conda env create -f environment.yml`

For Anaconda3
`conda env create -f environment3.yml`

Then activate the environment:

On Linux/MacOS:
`source activate neurocnn`

On Windows:
`activate neurocnn`

- add NEURON and (LFPy)

## Cell simulations:

Cell models can be downloaded from the Neocortical Micro Circuit Portal https://bbp.epfl.ch/nmc-portal/welcome
(13 models from layer 5 for testing are already included).

Newly downloaded models should be unzipped to the folder "cell_models/bbp/"
Then you must run "python hbp_cells.py compile" to compile .mod files. This only has to be done once,
if you do not add more models or manipulate the .mod files
(if you find problems in compiling try to install: sudo apt-get install lib32ncurses5-dev)

To run all L5 cell models you have downloaded, run "python do_all_cell_simulations.py"
From the command line you can give arguments to customize your simulated data:

- -rot -- 3d rotations to apply to the models befor computing EAPs (Norot-physrot-3drot)
- -intraonly -- only simulate intracellular dynamics (leave EAP simulation for later)
- -probe -- choose the probe type from the list in electrodes folder*
- -n -- number of observations per cell type

*or create your own! All you have to do is create a .json file with some specs and add you probe in the get_elcoords()
function of the MEAutility file

Your EAPs will be stored in the /spikes/bbp/'rotation_type'/ folder and the folder contaning the spikes is named:
e_n-obs-per-spike_1px_y-pitch_z-pitch_MEAname_date

## Localization and classification with CNNs (Tensorflow):

After EAPs are simulated you can train CNNs for localization and classification.

To run localization cd in the `localization` directory and run:

`python conv_localization.py -f path-to-spikes-folder`

You can give command line arguments to customize the network:

- -f full-path
- -feat -- feature type: Na - Rep - NaRep (default) - 3d (Waveform)
- -val -- validation: holdout (default) - hold-model-out
- -n -- number of training steps
- -cn -- cell model names to use
- -tcn -- train cellnames file*
- -vcn -- validation cellnames file*
- -s -- CNN size: xs - s - m - l (default) - xl
- -modelout -- model to hold out (in case validation is hold-model-out)
- -seed -- random seed to shuffle the data

To run classification cd in the `classification` directory and run:

`python conv_classification.py -f path-to-spikes-folder`

The command line arguments are the same as localization except:

- -feat -- feature type: AW - FW (default) - AFW - 3d (Waveform)
- -cl -- classification type: binary (excitatory-inhibitory - default) - m-type

*when -tcn and -vcn arguments are used, the CNN is trained on the models in the -tcn file and validated on the models in
the -vcn one (we recommend validating on cell models not used for training to avoid overfitting)

The models are stored in `localization(classification)/models/` and named:
model_rotation_feat-type_val-type_size_spike-folder_date

## Using the models

To reload the models and localize (or classify) another dataset you can use the `predict_location.py`
(or `predict_classification.py`) scripts.

`python predict_localization.py -mod CNN-model-folder -val validation-spikes-folder`

The `validation-spikes-folder` can be either a spikes folder generated with `do_all_cell_simulations.py` or a
CNN-model folder, which contains a folder named `validation_data` which contains the following .npy files:

- val_cat.npy -- category (neural type)
- val_loc.npy -- x,y,z location
- val_rot.npy -- 3d rotation of the cell model
- val_feat.npy -- features used for training
- val_spikes.npy -- EAP data
- val_mcat.npy (only for `predict_classification.py` -- cell m-type

The scripts computes the features depending on the CNN model used and outputs the accuracy on the validaiton dataset.

## Test scripts

Please take a look at the `test_script.sh` in the localization and classification folders to see the complete list of
commands to compile, simulate EAPs, train the CNNs, and validate the results.

## References

For details please take a look at our paper: "Combining forward modeling and deep learning for neural localization and
classification", ...

## Contact us

If you have problems running the code don't hesitate to contact us, or write an issue on the github page.

Alessio Buccino - alessiob@ifi.uio.no

Michael Kordovan - michael.kordovan@bcf.uni-freiburg.de
