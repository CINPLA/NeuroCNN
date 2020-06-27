'''

Deep net to classify Excit and Inhib neurons from features extracted from signals

'''
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, sys
from os.path import join
import shutil
import csv
import re
import time
import yaml
import pickle

base_folder = os.path.dirname(os.getcwd())
sys.path.insert(0, base_folder)

from defaultconfig import *
if os.path.exists(join(base_folder,'config.py')):
    from config import *

from tools import *
from tftools import *

TOWER_NAME = 'tower'

class SpikeConvNet:
    """ Convolutional Neural Network class for neuron classification
    based on extracellular action potentials (EAPs) measured with 
    Multi-Electrode Arrays (MEAs)

    Parameters:
    -----------
    train, bool (optional, default True)
        Whether to train the network
    noise_level, float (optional, default None)
        Standard deviation for additive Gaussian signal noise.
        If ``None``, no noise is added.
    save, bool (optional, default False)
        Whether to save the results
    keep_tmp_data, bool (optional, default False)
        Whether to keep the temporary training data
    spike_folder, str (optional, default None)
        Path to spike data used for training and validation. 
        If ``None``, an arbitrary spike folder of the data directory is taken.
    cellnames, str (optional, default 'all')
        Name of a file in the classification directory which specifies cell 
        names to be used. If ``all``, all available cells in the spike_folder
        directory are used. CAUTION: This affects training and learning and 
        cannot be used together with the train_cell_names and val_cell_names 
        parameters which should be used instead.
    train_cell_names, str (optional, default None)
        Name of a file in the classification directory which specifies cell 
        names to be used for training. If ``None``, cells are selected according
        to the cellnames parameter.
    val_cell_names, str (optional, default None)
        Name of a file in the classification directory which specifies cell 
        names to be used for validation. If ``None``, cells are selected
        according to the cellnames parameter.
    nsteps, int (optional, default=2000)
        Number of training epochs,
    n_spikes, int (optional, default None)
        Number of spikes taken into account. If ``None``, all available spikes
        are taken into account (in a balanced manner).
    val_percent, float (optional, default=10.)
        Percentage of data to be left out for validation in combination with
        val_type=``holdout``.
    feat_type, str (optional, defauls 'AW')
        String which specifies features of EAPs to be used by the CNN. Possible 
        values are 'AW','FW','AFW','3d'.
    val_type, str (optional, default 'holdout')
        Specifies the validation method. Possible values are 'holdout', 'k-fold'
        'hold-model-out', 'k-fold-model'. CAUTION: If 'train_cell_names' and 
        'val_cell_names' are specified, these completely specify the training
        and validation data splitting.
    kfolds, int (optional, default=5)
        Number of runs in a 'k-fold' validation setting.
    model_out, int (optional, default=5)
        Model number to be left out in a 'model-out' validation setting.
    size, str (optional, default 'l')
        Specifies CNN size. Possible values: 'xs','s','m','l','xl'
    class_type, str (optional, default 'm-type')
        Classification type ('binary' for excitatory/inhibitory classification
        or 'm-type' for morphological subclass classification.
    seed, int (optional, default=2308)
        Random seed for TensorFlow variable initialization. 
    """
    def __init__(self, train=True, noise_level=None, save=False,keep_tmp_data=False,
                 spike_folder = None, cellnames='all',
                 train_cell_names=None, val_cell_names=None, nsteps=2000,
                 n_spikes=None, val_percent=10.,
                 feat_type='AW', val_type = 'holdout', kfolds = 5, size='l', class_type='m-type',
                 model_out=5,seed=2308):
        """ Initialization
        """
        if not noise_level is None:
            self.noise_level = noise_level
        else:
            self.noise_level = 0  # uV
        self.train = train
        self.save = save
        self.seed = seed

        # If self.extra_overfit=True, an additional dropout in convolutional 
        # layers is implemented and variables in the fully connected layer have
        # additional weight decay to prevent from overfitting
        self.extra_overfit = False 

        # AW (amplitude-width)
        # FW (FWHM - Width)
        # AF (amplitude-FWHM)
        # AFW (amplitude-FHWM-width)
        # 3d (downsampled version of the entire waveform)
        self.feat_type = feat_type
        print('Feature type: ', self.feat_type)

        self.val_type = val_type
        if train_cell_names:
            self.val_type = 'provided-datasets'
        print('Validation type: ', self.val_type)
        if self.val_type == 'k-fold':
            self.kfolds = kfolds

        self.model_to_hold_out = model_out

        self.size = size
        print('Network size: ', self.size)

        # specify full path to dataset to be used containing simulated data from all cells
        if spike_folder is not None:
            self.spike_folder = spike_folder
        else:
            self.spike_folder = join(join(data_dir, 'spikes', os.listdir(join(data_dir, 'spikes'))[-1]),
                                     os.listdir(join(data_dir,'spikes',os.listdir(join(data_dir,'spikes'))[-1]))[0])
        self.cell_folder = join(root_folder, 'cell_models')

        s = self.spike_folder.split('/')
        # check if there is an ending backslash
        if s[-1] != '':
            self.spike_file_name = s[-1]
        else:
            self.spike_file_name = s[-2]

        # open 1 yaml file and save pitch
        yaml_files = [f for f in os.listdir(self.spike_folder) if '.yaml' in f or '.yml' in f]
        with open(join(self.spike_folder, yaml_files[0]), 'r') as f:
            self.info = yaml.load(f)

        self.rotation_type = self.info['Location']['rotation']
        print('Rotation type: ', self.rotation_type)
        self.pitch = self.info['Electrodes']['pitch']
        self.electrode_name = self.info['Electrodes']['electrode_name']
        print('Electrode name: ', self.electrode_name)
        self.mea_dim = self.info['Electrodes']['dim']
        self.n_elec = np.prod(self.info['Electrodes']['dim'])
        self.n_points = self.info['Electrodes']['n_points']
        self.spike_per_neuron = self.info['General']['target spikes']

        self.binary_cat = ['EXCIT', 'INHIB']

        if 'bbp' in self.spike_folder:
            self.all_categories = ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC',
                                   'NGC', 'SBC', 'STPC', 'TTPC1', 'TTPC2', 'UTPC']
            self.exc_categories = ['STPC', 'TTPC1', 'TTPC2', 'UTPC']
            self.inh_categories = ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC', 'NGC', 'SBC']

        self.categories_by_type = {'EXCIT': self.exc_categories,
                                   'INHIB': self.inh_categories}

        self.threshold_detect = 5  # uV
        self.classification = class_type  # binary/m-type

        # classification categories
        if self.classification == 'binary':
            self.class_categories = set() # should equal loaded categories to have no empty classes, gets filled while reading data
            self.pred_categories = self.binary_cat
            self.pred_dict = dict(zip(range(len(self.binary_cat)),self.binary_cat))
        else:
            self.class_categories = set() # should equal loaded categories to have no empty classes, gets filled while reading data
            self.pred_categories = set()
            self.pred_dict = {}
                    
        self.threshold_detect = 5  # uV

        self.all_etypes = ['cADpyr', 'cAC', 'bAC', 'cNAC', 'bNAC', 'dNAC', 'cSTUT', 'bSTUT', 'dSTUT', 'cIR', 'bIR']

        self.val_percent = val_percent

        if train_cell_names:
            self.train_cell_names = list(np.loadtxt(join(root_folder, 'classification', train_cell_names), dtype=str))
            print('cells from file used for training: ', self.train_cell_names)
            if val_cell_names:
                self.val_cell_names = list(np.loadtxt(join(root_folder, 'classification', val_cell_names), dtype=str))
                print('cells from file used for validation: ', self.val_cell_names)
            self.cell_names = self.train_cell_names + self.val_cell_names
        else:
            self.train_cell_names = None
            self.val_cell_names = None

            if cellnames == 'all':
                # self.cell_names = [f for f in os.listdir(self.cell_folder) if f.startswith('L5')]
                spikefiles = [f for f in os.listdir(self.spike_folder) if 'spikes' in f]
                self.cell_names = []
                for ss in spikefiles:
                    split_ss = ss.split('_')
                    self.cell_names.append('_'.join(split_ss[3:-1]))
                print('all cells used for training: ', self.cell_names)
            else:
                self.cell_names = list(np.loadtxt(join(root_folder, 'classification', cellnames), dtype=str))
                np.random.shuffle(self.cell_names)
                print('cells from file used for training: ', self.cell_names)

        self.dt = 2**-5

        # Setup Network parameters
        self.learning_rate = 5e-4
        self.training_epochs = nsteps
        self.batch_size = 1000
        self.display_step = 10
        self.dropout_rate = 0.7
           # Network Parameters
        if self.feat_type == 'AFW':
            self.inputs = 3
            self.ctsize = 0
        elif self.feat_type == 'FWRS':
            self.inputs = 4
            self.ctsize = 0
        elif self.feat_type == '3d':
            self.downsampling_factor = 4 # 4
            self.ctsize = 4
        else:
            self.inputs = 2
            self.ctsize = 0

        # Network size
        if self.size == 'xs':
            self.c1size = [2, 2]  # 1st layer number of features
            self.c2size = [2, 2]  # 2nd layer number of features
            self.l1depth = 4
            self.l2depth = 8
            self.fully = 128
        elif self.size == 's':
            self.c1size = [2, 2]  # 1st layer number of features
            self.c2size = [2, 2]  # 2nd layer number of features
            self.l1depth = 8
            self.l2depth = 16
            self.fully = 256
        elif self.size == 'm':
            self.c1size = [3, 3]  # 1st layer number of features
            self.c2size = [3, 3]  # 2nd layer number of features
            self.l1depth = 16
            self.l2depth = 32
            self.fully = 512
        elif self.size == 'l':
            self.c1size = [3, 3]  # 1st layer number of features
            self.c2size = [3, 3]  # 2nd layer number of features
            self.l1depth = 32
            self.l2depth = 64
            self.fully = 1024
        elif self.size == 'xl':
            self.c1size = [3, 3]  # 1st layer number of features
            self.c2size = [3, 3]  # 2nd layer number of features
            self.l1depth = 64
            self.l2depth = 128
            self.fully = 2048

        if n_spikes and type(n_spikes) is int:
            self.n_spikes = n_spikes
        else:
            self.n_spikes = 'all'


        # create model folder
        self.model_name = 'model_' + self.rotation_type + '_' + self.classification + '_' +\
                     self.feat_type + '_' + self.val_type + \
                     '_' + self.size + '_' + self.spike_file_name + '_' + \
                     time.strftime("%d-%m-%Y:%H:%M")
        class_data_dir = join(data_dir,'classification')
        self.model_path = join(class_data_dir, 'models', self.model_name)
        os.makedirs(self.model_path)

        self.cell_dict = {}

        self.val_data_dir = join(self.model_path, 'validation_data')
        if not os.path.isdir(self.val_data_dir):
            os.makedirs(self.val_data_dir)

        self.ndatafiles = 10
        self.datacounter = 0
        self.datafilenames = ['data_batch_%d.csv' % fn for fn in range(self.ndatafiles) ]

        ######################
        # data preprocessing #
        ######################
        self._data_preprocessing(self.all_categories)
        
        self.class_categories = list(self.class_categories)
        self.pred_categories = list(self.pred_categories)
        self.num_categories = len(self.pred_categories)
        self.out = self.num_categories

        ############
        # training #
        ############
        if self.train:
            t_start = time.time()

            self.accuracies, self.cat, self.loc, self.rot, self.guessed = [], [], [], [], []

            if self.validation_runs > 1:
                for v in range(self.validation_runs):
                    print('Training/Validation run: ', v+1, '/', self.validation_runs)
                    self.training(validation_run=v)
                    self.evaluate(validation_run=v)
            else:
                self.training()
                self.evaluate()

            self.accuracies = np.squeeze(np.array(self.accuracies))
            self.loc = np.squeeze(np.array(self.loc))
            self.rot = np.squeeze(np.array(self.rot))
            self.cat = np.squeeze(np.array(self.cat))
            self.guessed = np.squeeze(np.array(self.guessed))

            print("Training done")
            t_end = time.time()
            self.processing_time = t_end - t_start
            print('Training time: ', self.processing_time)

            if self.save:
                self.save_meta_model()

        # cleaning up
        if not keep_tmp_data:
            self.remondis()


    def _data_preprocessing(self, categories, balanced=True):
        ''' Data preprocessing: 
        - organize training and validation data
        - extract the features
        - put everything into temporary directions, 
          from which an input queue is created in tensorflow
        
        Parameters:
        -----------
        categories, list
            List of cell categories to load
        balanced, bool (optional, default True)
            Whether to balance the data according to predicted classes
        '''
        if not self.train_cell_names:
            # Get even number of observations per class
            spikelist = [f for f in os.listdir(self.spike_folder)
                         if f.startswith('e_spikes') and  any(x in f for x in self.cell_names)]
            cells, occurrences = np.unique(sum([[f.split('_')[4]]*int(f.split('_')[2]) for f in spikelist],[]),
                                           return_counts=True)

            occ_dict = dict(zip(cells,occurrences))
            occ_exc = sum([occ_dict[cc] for cc in cells if cc in self.exc_categories])
            occ_inh = sum([occ_dict[cc] for cc in cells if cc in self.inh_categories])
            binary_occ = [occ_exc, occ_inh]

            min_binary_occ = np.min(binary_occ)
            min_mtype_occ = np.min(occurrences)

            if self.n_spikes is not 'all':
                min_binary_occ = np.min(self.n_spikes,min_binary_occ)
                min_mtype_occ = np.min(self.n_spikes,min_mtype_occ)
                print('Changed min_occurencies due to self.n_spikes')

            print('MIN BINARY OCCURRENCE: ', min_binary_occ)
            print('MIN M-TYPE OCCURRENCE: ', min_mtype_occ)

            self.morph_ids = np.unique([int(f.split('_')[6]) for f in spikelist])
            self.num_morph = len(self.morph_ids)
            if self.val_type == 'hold-model-out':
                if self.num_morph > 1:
                    # self.model_out = np.random.randint(1, self.num_morph)
                    self.model_out = self.model_to_hold_out
                    print('Hold model out: ', self.model_out)
                else:
                    raise AttributeError('Hold-model-out can be run when more than one cell id is in the dataset')

            # set up temporary directories
            self.tmp_data_dir = join(data_dir,'classification','tmp'+time.strftime("%d-%m-%Y:%H:%M")+
                                     str(np.random.rand()).split('.')[1])
            if not os.path.isdir(self.tmp_data_dir):
                os.makedirs(self.tmp_data_dir)
            else:
                raise IOError('Temporary data directory exists already.')

            self.num_train_feat = 0
            self.num_val_feat = 0

            for idx, cell in enumerate(categories):
                avail_samples = sum([int(f.split('_')[2]) for f in spikelist if cell in f])
                print(idx+1, '/', len(categories), ' :', cell, ' avail: ', avail_samples)
                cells_to_load = [c for c in self.cell_names if cell in c]

                ncells_inh = len([c for c in cells if c in self.inh_categories])
                ncells_exc = len([c for c in cells if c in self.exc_categories])
                min_inh_occ = int(float(min_binary_occ)/ncells_inh)
                min_exc_occ = int(float(min_binary_occ)/ncells_exc)

                spikes, loc, rot, cat, etype, morphid, loaded_cat = load_EAP_data(self.spike_folder, \
                                                                                  cells_to_load, categories)
                # subsample min_occurrences
                if balanced:
                    if self.classification == 'binary':
                        if cell in self.exc_categories:
                            if len(cat) > min_exc_occ:
                                rand_per = np.random.permutation(len(cat))
                                idx_cells = rand_per[:min_exc_occ]
                        elif cell in self.inh_categories:
                            if len(cat) > min_inh_occ:
                                rand_per = np.random.permutation(len(cat))
                                idx_cells = rand_per[:min_inh_occ]
                            spikes = spikes[idx_cells]
                            loc = loc[idx_cells]
                            rot = rot[idx_cells]
                            cat = cat[idx_cells]
                            etype = etype[idx_cells]
                            morphid = morphid[idx_cells]
                    else:
                        if len(cat) > min_mtype_occ:
                            rand_per = np.random.permutation(len(cat))
                            idx_cells  = rand_per[:min_mtype_occ]
                            spikes = spikes[idx_cells]
                            loc = loc[idx_cells]
                            rot = rot[idx_cells]
                            cat = cat[idx_cells]
                            etype = etype[idx_cells]
                            morphid = morphid[idx_cells]


                if len(cat) > 0:
                    # classification categories
                    if self.classification == 'binary':
                        mcat = self.get_mtype_cat_idx(cat, cells)
                        cat = self.get_binary_cat_idx(cat)
                        self.cell_dict.update({int(np.unique(mcat)): cell})
                        self.class_categories.update(loaded_cat)
                    else:
                        cat = self.get_mtype_cat_idx(cat, cells)
                        mcat = None
                        self.cell_dict.update({int(np.unique(cat)): cell})
                        self.pred_dict.update({int(np.unique(cat)): cell})
                        self.class_categories.update(loaded_cat)
                        self.pred_categories.update(loaded_cat)

                    ## convert etype to integer
                    for i,typ in enumerate(etype):
                        for j,et in enumerate(self.all_etypes):
                            if et in typ:
                                etype[i] = j

                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(spikes.shape[0], spikes.shape[1], spikes.shape[2]))
                        spikes += noise

                    # divide train an validation data
                    self.create_tmp_data(spikes,loc,rot,cat,etype,morphid,mcat=mcat)
        else:
            self.validation_runs = 1
            # Load and balance training and val separately
            # Get even number of observations per class
            train_spikelist = [f for f in os.listdir(self.spike_folder) if
                               f.startswith('e_spikes') and any(x in f for x in self.train_cell_names)]

            train_cells, train_occurrences = np.unique(
                sum([[f.split('_')[4]] * int(f.split('_')[2]) for f in train_spikelist], []),
                return_counts=True)
            train_occ_dict = dict(zip(train_cells, train_occurrences))
            train_min_occurrence = np.min(train_occurrences)

            train_occ_exc = sum([train_occ_dict[cc] for cc in train_cells if cc in self.exc_categories])
            train_occ_inh = sum([train_occ_dict[cc] for cc in train_cells if cc in self.inh_categories])
            train_binary_occ = [train_occ_exc, train_occ_inh]
            train_min_binary_occ = np.min(train_binary_occ)
            train_min_mtype_occ = np.min(train_occurrences)

            val_spikelist = [f for f in os.listdir(self.spike_folder) if
                             f.startswith('e_spikes') and any(x in f for x in self.val_cell_names)]
            val_cells, val_occurrences = np.unique(
                sum([[f.split('_')[4]] * int(f.split('_')[2]) for f in val_spikelist], []),
                return_counts=True)
            val_occ_dict = dict(zip(val_cells, val_occurrences))
            val_min_occurrence = np.min(val_occurrences)
            val_occ_exc = sum([val_occ_dict[cc] for cc in val_cells if cc in self.exc_categories])
            val_occ_inh = sum([val_occ_dict[cc] for cc in val_cells if cc in self.inh_categories])
            val_binary_occ = [val_occ_exc, val_occ_inh]
            val_min_binary_occ = np.min(val_binary_occ)
            val_min_mtype_occ = np.min(val_occurrences)

            if self.n_spikes is not 'all':
                train_min_binary_occ = np.min(self.n_spikes, train_min_binary_occ)
                train_min_mtype_occ = np.min(self.n_spikes, train_min_mtype_occ)
                val_min_binary_occ = np.min(self.n_spikes, val_min_binary_occ)
                val_min_mtype_occ = np.min(self.n_spikes, val_min_mtype_occ)
                print('Changed min_occurencies due to self.n_spikes')
            print('MIN BINARY OCCURRENCE: train - ', train_min_binary_occ, ' val - ', val_min_binary_occ)
            print('MIN M-TYPE OCCURRENCE: train - ', train_min_mtype_occ, ' val - ', val_min_mtype_occ)

            # set up temporary directories
            self.tmp_data_dir = join(data_dir, 'classification',
                                     'tmp' + time.strftime("%d-%m-%Y:%H:%M") + str(np.random.rand()).split('.')[1])

            if not os.path.isdir(self.tmp_data_dir):
                os.makedirs(self.tmp_data_dir)
            else:
                raise IOError('Temporary data directory exists already.')

            self.num_train_feat = 0
            self.num_val_feat = 0

            # do loading and feature extraction for each cell separately
            for idx, cell in enumerate(categories):
                train_avail_samples = sum([int(f.split('_')[2]) for f in train_spikelist if cell in f])

                train_n_samples = train_min_occurrence  # int(avail_samples / occ_dict[cell] * min_occurrence)
                train_cells_to_load = [c for c in self.train_cell_names if '_' + cell + '_' in c]
                train_spikes, train_loc, train_rot, train_cat, train_etype, train_morphid, loaded_cat = \
                    load_EAP_data(self.spike_folder, train_cells_to_load, categories)

                ncells_inh = len([c for c in train_cells if c in self.inh_categories])
                ncells_exc = len([c for c in train_cells if c in self.exc_categories])
                train_min_inh_occ = int(float(train_min_binary_occ) / ncells_inh)
                train_min_exc_occ = int(float(train_min_binary_occ) / ncells_exc)

                # subsample min_occurrences
                if balanced:
                    if self.classification == 'binary':
                        if cell in self.exc_categories:
                            if len(train_cat) > train_min_exc_occ:
                                rand_per = np.random.permutation(len(train_cat))
                                idx_cells = rand_per[:train_min_exc_occ]
                                train_spikes = train_spikes[idx_cells]
                                train_loc = train_loc[idx_cells]
                                train_rot = train_rot[idx_cells]
                                train_cat = train_cat[idx_cells]
                                train_etype = train_etype[idx_cells]
                                train_morphid = train_morphid[idx_cells]
                        elif cell in self.inh_categories:
                            if len(train_cat) > train_min_inh_occ:
                                rand_per = np.random.permutation(len(train_cat))
                                idx_cells = rand_per[:train_min_inh_occ]
                                train_spikes = train_spikes[idx_cells]
                                train_loc = train_loc[idx_cells]
                                train_rot = train_rot[idx_cells]
                                train_cat = train_cat[idx_cells]
                                train_etype = train_etype[idx_cells]
                                train_morphid = train_morphid[idx_cells]
                    else:
                        if len(train_cat) > train_min_mtype_occ:
                            rand_per = np.random.permutation(len(train_cat))
                            idx_cells = rand_per[:train_min_mtype_occ]
                            train_spikes = train_spikes[idx_cells]
                            train_loc = train_loc[idx_cells]
                            train_rot = train_rot[idx_cells]
                            train_cat = train_cat[idx_cells]
                            train_etype = train_etype[idx_cells]
                            train_morphid = train_morphid[idx_cells]
                n_train = len(train_cat)

                if len(train_cat) > 0:
                    # classification categories
                    if self.classification == 'binary':
                        train_mcat = self.get_mtype_cat_idx(train_cat, train_cells)
                        train_cat = self.get_binary_cat_idx(train_cat)
                        self.cell_dict.update({int(np.unique(train_mcat)): cell})
                        self.class_categories.update(loaded_cat)
                    else:
                        train_cat = self.get_mtype_cat_idx(train_cat, train_cells)
                        train_mcat = None
                        self.cell_dict.update({int(np.unique(train_cat)): cell})
                        self.pred_dict.update({int(np.unique(train_cat)): cell})
                        self.class_categories.update(loaded_cat)
                        self.pred_categories.update(loaded_cat)

                    # # convert etype to integer
                    for i, typ in enumerate(train_etype):
                        for j, et in enumerate(self.all_etypes):
                            if et in typ:
                                train_etype[i] = j

                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(train_spikes.shape[0], train_spikes.shape[1],
                                                       train_spikes.shape[2]))
                        train_spikes += noise

                    # divide train an validation data
                    self.create_tmp_data(train_spikes, train_loc, train_rot, train_cat, train_etype, train_morphid,
                                         mcat=train_mcat, dset='train')

                    del train_spikes, train_loc, train_rot, train_cat, train_etype, train_morphid

                val_avail_samples = sum([int(f.split('_')[2]) for f in val_spikelist if cell in f])
                val_spikelist = [f for f in os.listdir(self.spike_folder) if
                                 f.startswith('e_spikes') and any(x in f for x in self.val_cell_names)]
                val_cells, val_occurrences = np.unique(
                    sum([[f.split('_')[4]] * int(f.split('_')[2]) for f in val_spikelist], []),
                    return_counts=True)
                val_occ_dict = dict(zip(val_cells, val_occurrences))
                val_min_occurrence = np.min(val_occurrences)
                val_n_samples = val_min_occurrence  # int(avail_samples / occ_dict[cell] * min_occurrence)
                val_cells_to_load = [c for c in self.val_cell_names if '_' + cell + '_' in c]
                val_spikes, val_loc, val_rot, val_cat, val_etype, val_morphid, loaded_cat = \
                    load_EAP_data(self.spike_folder, val_cells_to_load, categories)
                print(idx + 1, '/', len(categories), ' :', cell, ' avail train: ', train_avail_samples, \
                    ' avail val: ', val_avail_samples)

                ncells_inh = len([c for c in val_cells if c in self.inh_categories])
                ncells_exc = len([c for c in val_cells if c in self.exc_categories])
                val_min_inh_occ = int(float(val_min_binary_occ) / ncells_inh)
                val_min_exc_occ = int(float(val_min_binary_occ) / ncells_exc)

                # subsample min_occurrences
                if balanced:
                    if self.classification == 'binary':
                        if cell in self.exc_categories:
                            if len(val_cat) > val_min_exc_occ:
                                rand_per = np.random.permutation(len(val_cat))
                                idx_cells = rand_per[:val_min_exc_occ]
                                val_spikes = val_spikes[idx_cells]
                                val_loc = val_loc[idx_cells]
                                val_rot = val_rot[idx_cells]
                                val_cat = val_cat[idx_cells]
                                val_etype = val_etype[idx_cells]
                                val_morphid = val_morphid[idx_cells]
                        elif cell in self.inh_categories:
                            if len(val_cat) > val_min_inh_occ:
                                rand_per = np.random.permutation(len(val_cat))
                                idx_cells = rand_per[:val_min_inh_occ]
                                val_spikes = val_spikes[idx_cells]
                                val_loc = val_loc[idx_cells]
                                val_rot = val_rot[idx_cells]
                                val_cat = val_cat[idx_cells]
                                val_etype = val_etype[idx_cells]
                                val_morphid = val_morphid[idx_cells]
                    else:
                        if len(val_cat) > val_min_mtype_occ:
                            rand_per = np.random.permutation(len(val_cat))
                            idx_cells = rand_per[:val_min_mtype_occ]
                            val_spikes = val_spikes[idx_cells]
                            val_loc = val_loc[idx_cells]
                            val_rot = val_rot[idx_cells]
                            val_cat = val_cat[idx_cells]
                            val_etype = val_etype[idx_cells]
                            val_morphid = val_morphid[idx_cells]
                n_val = len(val_cat)

                if len(val_cat) > 0:
                    # classification categories
                    if self.classification == 'binary':
                        val_mcat = self.get_mtype_cat_idx(val_cat, val_cells)
                        val_cat = self.get_binary_cat_idx(val_cat)
                        self.cell_dict.update({int(np.unique(val_mcat)): cell})
                        self.class_categories.update(loaded_cat)
                    else:
                        val_cat = self.get_mtype_cat_idx(val_cat, val_cells)
                        val_mcat = None
                        self.cell_dict.update({int(np.unique(val_cat)): cell})
                        self.pred_dict.update({int(np.unique(val_cat)): cell})
                        self.class_categories.update(loaded_cat)
                        self.pred_categories.update(loaded_cat)

                    # # convert etype to integer
                    for i, typ in enumerate(val_etype):
                        for j, et in enumerate(self.all_etypes):
                            if et in typ:
                                val_etype[i] = j
                    #
                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(val_spikes.shape[0], val_spikes.shape[1], val_spikes.shape[2]))
                        val_spikes += noise
                    # divide train an validation data
                    self.create_tmp_data(val_spikes, val_loc, val_rot, val_cat, val_etype, val_morphid,
                                         mcat=val_mcat, dset='validation')


    def return_features(self,spikes):
        ''' Extract features from spikes

        Parameters:
        -----------
        spikes, array_like
            Spike data to extract features from

        Returns:
        --------
        features, array_like
        '''
        # Create feature set:
        if self.feat_type == 'AW':
            features = get_EAP_features(spikes,['A','W'],\
                                             dt=self.dt,threshold_detect=self.threshold_detect)
            features = np.array([features['amps'],features['widths']])
            self.inputs = 2  # amps and widths
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == 'FW':
            features = get_EAP_features(spikes,['F','W'],\
                                             dt=self.dt,threshold_detect=self.threshold_detect)
            features = np.array([features['fwhm'],features['widths']])
            self.inputs = 2  # fwhm and widths
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == 'AF':
            features = get_EAP_features(spikes,['A','F'],\
                                             dt=self.dt,threshold_detect=self.threshold_detect)
            features = np.array([features['amps'],features['fwhm']])
            self.inputs = 2  # amps and fwhm
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == 'AFW':
            features = get_EAP_features(spikes,['A','F','W'],\
                                             dt=self.dt,threshold_detect=self.threshold_detect)
            features = np.array([features['amps'],features['fwhm'],features['widths']])
            self.inputs = 3  # amps, fwhms and widths
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == 'FWRS':
            features = get_EAP_features(spikes,['F','W', 'R', 'S'],\
                                             dt=self.dt,threshold_detect=self.threshold_detect)
            features = np.array([features['fwhm'],features['widths'],features['ratio'],features['speed']])
            self.inputs = 4  # fwhms, widths, ratios, speed
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == '3d':
            print('Downsampling spikes...')
            # downsampled_spikes = ss.resample(self.spikes, self.spikes.shape[2] // self.downsampling_factor, axis=2)
            downsampled_spikes = spikes[:,:,::int(self.downsampling_factor)]
            features = downsampled_spikes
            self.inputs = spikes.shape[2] // self.downsampling_factor
            print('Done')

        return features

    def create_tmp_data(self,spikes,loc,rot,cat,etype,morphid,mcat=None,dset=None):
        ''' Divide train and validation data and store it temporary
        
        Parameters:
        -----------
        spikes, array_like
        loc, array_like
        rot, array_like
        cat, array_like
        etype, array_like
        morphid, array_like
        mcat, array_like (optional, default None)
        dset, str (optional, default None)
            If ``None``, parameter arrays are split according to self.val_type .
            If 'train', all input data is treaten as training data,
            if 'validation', input data is treaten as validation data.
        '''
        if not dset:
            num_spikes = len(list(cat))
            validation_mcat=None

            # shuffle observaitons:
            shuffle_idx = np.random.permutation(num_spikes)
            spikes = spikes[shuffle_idx]
            loc = loc[shuffle_idx]
            rot = rot[shuffle_idx]
            cat = cat[shuffle_idx]
            etype = etype[shuffle_idx]
            morphid = morphid[shuffle_idx]
            if mcat is not None:
                mcat = mcat[shuffle_idx]

            if self.val_type == 'holdout':
                self.validation_runs = 1
                tmp_train_dir=join(self.tmp_data_dir,'train')
                curr_eval_dir=self.val_data_dir
                if not os.path.isdir(tmp_train_dir):
                    os.makedirs(tmp_train_dir)

                validation_size = int(num_spikes / self.val_percent)
                train_spikes = spikes[:-validation_size]
                train_feat = self.return_features(train_spikes)
                train_loc = loc[:-validation_size]
                train_rot = rot[:-validation_size]
                train_cat = cat[:-validation_size]
                train_etype = etype[:-validation_size]

                validation_spikes = spikes[-validation_size:]
                validation_feat = self.return_features(spikes[-validation_size:])
                validation_loc = loc[-validation_size:]
                validation_rot = rot[-validation_size:]
                validation_cat = cat[-validation_size:]
                if mcat is not None:
                    validation_mcat = mcat[-validation_size:]
                validation_etype = etype[-validation_size:]

                self.num_train_feat += len(train_cat)
                self.num_val_feat += len(validation_cat)

                self.dump_learndata(tmp_train_dir,[train_loc,train_rot,train_cat,train_etype,train_feat])
                self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                  validation_rot, validation_cat,mcat=validation_mcat)

            elif self.val_type == 'k-fold':
                self.validation_runs = self.kfolds
                subsample_size = int(num_spikes/self.kfolds)
                for kk in range(self.kfolds):
                    tmp_train_dir=join(self.tmp_data_dir,'train_'+str(kk))
                    curr_eval_dir=join(self.val_data_dir,'eval_'+str(kk))
                    if not os.path.isdir(curr_eval_dir):
                        os.makedirs(curr_eval_dir)
                    if not os.path.isdir(tmp_train_dir):
                        os.makedirs(tmp_train_dir)

                    train_idx = np.append(np.arange(0, kk * subsample_size),
                                          np.arange((kk + 1) * subsample_size, len(spikes)))
                    val_idx = np.arange(kk * subsample_size, (kk + 1) * subsample_size)

                    train_spikes = spikes[train_idx]
                    train_feat = self.return_features(train_spikes)
                    train_loc = loc[train_idx]
                    train_rot = rot[train_idx]
                    train_cat = cat[train_idx]
                    train_etype = etype[train_idx]

                    validation_spikes = spikes[val_idx]
                    validation_feat = self.return_features(spikes[val_idx])
                    validation_loc = loc[val_idx]
                    validation_rot = rot[val_idx]
                    validation_cat = cat[val_idx]
                    if mcat is not None:
                        validation_mcat = mcat[val_idx]
                    validation_etype = etype[val_idx]


                    self.dump_learndata(tmp_train_dir,[train_loc,train_rot,train_cat,train_etype,train_feat])
                    self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                      validation_rot, validation_cat,mcat=validation_mcat)

                self.num_train_feat += len(train_cat)
                self.num_val_feat += len(validation_cat)

            elif self.val_type == 'hold-model-out':
                self.validation_runs = 1
                tmp_train_dir=join(self.tmp_data_dir,'train')
                curr_eval_dir=self.val_data_dir
                if not os.path.isdir(tmp_train_dir):
                    os.makedirs(tmp_train_dir)

                if self.num_morph > 1:
                    train_idx = np.where(morphid != self.model_out)
                    val_idx = np.where(morphid == self.model_out)

                    if len(train_idx[0])!=0:
                        train_spikes = spikes[train_idx]
                        train_feat = self.return_features(train_spikes)
                        train_loc = loc[train_idx]
                        train_rot = rot[train_idx]
                        train_cat = cat[train_idx]
                        train_etype = etype[train_idx]
                        self.num_train_feat += len(train_cat)
                        self.dump_learndata(tmp_train_dir,[train_loc,train_rot,train_cat,train_etype,train_feat])

                    if len(val_idx[0])!=0:
                        validation_spikes = spikes[val_idx]
                        validation_feat = self.return_features(spikes[val_idx])
                        validation_loc = loc[val_idx]
                        validation_rot = rot[val_idx]
                        validation_cat = cat[val_idx]
                        if mcat is not None:
                            validation_mcat = mcat[val_idx]
                        validation_etype = etype[val_idx]
                        self.num_val_feat += len(validation_cat)
                        self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                          validation_rot, validation_cat,mcat=validation_mcat)
                else:
                    raise ValueError(
                        'Cross model validation can be performed only when multiple cell morphologies of same '
                        'me-type are availabel in the dataset.')

            elif self.val_type == 'k-fold-model':
                self.validation_runs = len(self.morph_ids)
                for kk in range(len(self.morph_ids)):
                    tmp_train_dir=join(self.tmp_data_dir,'train_'+str(kk))
                    curr_eval_dir=join(self.val_data_dir,'eval_'+str(kk))
                    if not os.path.isdir(curr_eval_dir):
                        os.makedirs(curr_eval_dir)
                    if not os.path.isdir(tmp_train_dir):
                        os.makedirs(tmp_train_dir)

                    train_idx = np.where(morphid != self.morph_ids[kk])
                    val_idx = np.where(morphid == self.morph_ids[kk])

                    if len(train_idx[0])!=0:
                        train_spikes = spikes[train_idx]
                        train_feat = self.return_features(train_spikes)
                        train_loc = loc[train_idx]
                        train_rot = rot[train_idx]
                        train_cat = cat[train_idx]
                        train_etype = etype[train_idx]
                        if kk==0:
                            self.num_train_feat += len(train_cat)
                        self.dump_learndata(tmp_train_dir,[train_loc,train_rot,train_cat,train_etype,train_feat])

                    if len(val_idx[0])!=0:
                        validation_spikes = spikes[val_idx]
                        validation_feat = self.return_features(spikes[val_idx])
                        validation_loc = loc[val_idx]
                        validation_rot = rot[val_idx]
                        validation_cat = cat[val_idx]
                        if mcat is not None:
                            validation_mcat = mcat[val_idx]
                        validation_etype = etype[val_idx]
                        if kk==0:
                            self.num_val_feat += len(validation_cat)
                        self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                          validation_rot, validation_cat,mcat=validation_mcat)
        elif dset == 'train':
            tmp_train_dir = join(self.tmp_data_dir, 'train')
            if not os.path.isdir(tmp_train_dir):
                os.makedirs(tmp_train_dir)
            num_spikes = len(list(cat))
            # shuffle observaitons:
            shuffle_idx = np.random.permutation(num_spikes)
            spikes = spikes[shuffle_idx]
            loc = loc[shuffle_idx]
            rot = rot[shuffle_idx]
            cat = cat[shuffle_idx]
            etype = etype[shuffle_idx]
            morphid = morphid[shuffle_idx]

            tmp_train_dir = join(self.tmp_data_dir, 'train')
            curr_eval_dir = self.val_data_dir

            if not os.path.isdir(tmp_train_dir):
                os.makedirs(tmp_train_dir)

            train_spikes = spikes

            train_feat = self.return_features(train_spikes)
            train_loc = loc
            train_rot = rot
            train_cat = cat
            train_etype = etype

            self.num_train_feat += len(train_cat)

            self.dump_learndata(tmp_train_dir, [train_loc, train_rot, train_cat, train_etype, train_feat])

        elif dset == 'validation':
            validation_mcat = None
            curr_eval_dir = self.val_data_dir
            if not os.path.isdir(curr_eval_dir):
                os.makedirs(curr_eval_dir)
            if mcat is not None:
                validation_mcat = mcat

            validation_spikes = spikes

            validation_feat = self.return_features(validation_spikes)
            validation_loc = loc
            validation_rot = rot
            validation_cat = cat
            validation_etype = etype
            self.num_val_feat += len(validation_cat)

            self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                              validation_rot, validation_cat, mcat=validation_mcat)


    def dump_learndata(self,filedir,data):
        """ dump the data to files in filedir
        
        Parameters:
        -----------
        filedir : string
            Path to directory the data is saved to.
        data : array_like
            List of data arrays to be saved.
        """

        for i in range(data[0].shape[0]):
            filename = self.datafilenames[self.datacounter % self.ndatafiles]
            outfile = join(filedir,filename)
            dataline = np.hstack([d[i].flatten() for d in data])
            # check existence
            if not os.path.exists(outfile):
                with open(outfile,'wb') as csvfile:
                    csv_writer = csv.writer(csvfile) # specify alternative delimiters, etc here
                    csv_writer.writerow(dataline)
            else:
                with open(outfile,'a+b') as csvfile:
                    csv_writer = csv.writer(csvfile) # specify alternative delimiters, etc here
                    csv_writer.writerow(dataline)
            self.datacounter += 1

    def dump_valdata(self, val_dir, spikes, feat, loc, rot, cat, mcat=None):
        """ dump validation data in *.npy files
        
        Parameters:
        -----------
        val_dir : string
            Path to the validation diretory where the data is saved to.
        spikes : array_like
            Spike data
        feat : array_like
            Spike features
        loc : array_like
            Locations of neuron somas.
        rot : array_like
            Array of neuron rotations.
        cat : array_like
            Array of neuron categories.
        mcat : array_like (optional, default=None)
            Morphological category of the neurons.
        """
        if not os.path.isfile(join(val_dir, 'val_spikes.npy')):
            np.save(join(val_dir, 'val_spikes'), spikes)
            np.save(join(val_dir, 'val_feat'), feat)
            np.save(join(val_dir, 'val_loc'), loc)
            np.save(join(val_dir, 'val_rot'), rot)
            np.save(join(val_dir, 'val_cat'), cat)
            if mcat is not None:
                np.save(join(val_dir, 'val_mcat'), mcat)
        else:
            np.save(join(val_dir, 'val_spikes'), np.vstack((np.load(join(val_dir, 'val_spikes.npy')), spikes)))
            np.save(join(val_dir, 'val_feat'), np.vstack((np.load(join(val_dir, 'val_feat.npy')), feat)))
            np.save(join(val_dir, 'val_loc'), np.vstack((np.load(join(val_dir, 'val_loc.npy')), loc)))
            np.save(join(val_dir, 'val_rot'), np.vstack((np.load(join(val_dir, 'val_rot.npy')), rot)))
            np.save(join(val_dir, 'val_cat'), np.concatenate((np.load(join(val_dir, 'val_cat.npy')), cat)))
            if mcat is not None:
                np.save(join(val_dir, 'val_mcat'), np.concatenate((np.load(join(val_dir, 'val_mcat.npy')), mcat)))

    def read_single_example(self,filename_queue):
        """ get a single example from filename queue

        Parameters:
        -----------
        filename_queue : TensorFlow Queue
            TensorFlow filename queue to read data from.
        
        Returns:
        --------
        result : object
            A result object with attributes loaded from a file in the queue.
        """
        class NclassRecord(object):
            pass
        result = NclassRecord()

        # data specific parameters
        label_sizes = [3,3,1,1] # loc,rot,cat,etype
        feature_size = self.n_elec * self.inputs
        record_size = np.sum(label_sizes) + feature_size

        # Create Reader
        reader = tf.TextLineReader()
        result.key , value = reader.read(filename_queue)

        # data preprocessing
        record_defaults = list(np.zeros((record_size,1),dtype='float32'))
        record_size = tf.decode_csv(value,record_defaults=record_defaults)

        result.loclabel = tf.cast(tf.slice(record_size, [0], [3]), tf.float32)
        result.rotlabel = tf.cast(tf.slice(record_size, [3], [3]), tf.float32)
        result.mlabel = tf.cast(tf.slice(record_size, [6], [1]), tf.int64)
        result.elabel = tf.cast(tf.slice(record_size, [7], [1]), tf.int64)
        # label as one hot

        result.mlabel = tf.SparseTensor([result.mlabel],[1.],dense_shape=[self.num_categories])
        result.mlabel = tf.sparse_tensor_to_dense(result.mlabel)

        result.features = tf.reshape(tf.slice(record_size, [np.sum(label_sizes)], [feature_size]),
                                       [self.n_elec,self.inputs])

        return result


    def inputs_fct(self,ddir,batch):
        ''' Returns a training batch of the data
        
        Parameters:
        -----------
        ddir : string
            Path to data directory.
        batch : int
            Batch size.

        Returns:
        --------
        feature_batch : TensorFlow Tensor
            Batch tensor of features.
        mlabel_batch : TensorFlow Tensor
            Batch tensor of labels
        '''
        # filenames
        filenames = [join(ddir,f) for f in os.listdir(ddir) if f.endswith('.csv')]
        # filename queue
        filename_queue = tf.train.string_input_producer(filenames,shuffle=True)

        # read one example from filename queue
        read_input = self.read_single_example(filename_queue)
        features = read_input.features
        mlabel = read_input.mlabel

        # generate the batch
        feature_batch, mlabel_batch = tf.train.shuffle_batch([features,mlabel],
                                                    batch_size=batch,
                                                    num_threads=2,
                                                    capacity=65000,
                                                    min_after_dequeue=30000)
        return feature_batch, mlabel_batch

    def inference(self,xx):
        ''' infer network prediction
        Parameters:
        -----------
        xx: tensor, graph input
        
        Returns:
        --------
        pred: tensor, prediction
        '''
        if self.feat_type != '3d':
            self.keep_prob = tf.placeholder("float",name='keep_prob')
            x_image = tf.reshape(xx, [-1, self.mea_dim[0], self.mea_dim[1], self.inputs])
            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([self.c1size[0], self.c1size[1], self.inputs, self.l1depth], "wconv1", self.seed)
                b_conv1 = bias_variable([self.l1depth], "wb1")
                h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1,name=scope.name)
                activation_summary(h_conv1,TOWER_NAME)
            if self.extra_overfit:
                h_conv1_drop = tf.nn.dropout(h_conv1, self.keep_prob)
                h_pool1 = max_pool_2d(h_conv1_drop)
            else:
                h_pool1 = max_pool_2d(h_conv1)
            # h_pool1_norm = tf.clip_by_norm(h_pool1,3.5)
            spatial_feat_size = np.ceil(np.array(self.mea_dim)/2.)
            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([self.c2size[0], self.c2size[1], self.l1depth, self.l2depth], "wconv2", self.seed)
                b_conv2 = bias_variable([self.l2depth], "wb2")
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,name=scope.name)
                activation_summary(h_conv2,TOWER_NAME)
            if self.extra_overfit:
                h_conv2_drop = tf.nn.dropout(h_conv2, self.keep_prob)
                h_pool2 = max_pool_2d(h_conv2_drop)
            else:
                h_pool2 = max_pool_2d(h_conv2)
            # h_pool2_norm = tf.clip_by_norm(h_pool2,3.5)
            spatial_feat_size = np.array(np.ceil(spatial_feat_size/2.),dtype=int)
            with tf.variable_scope('local3') as scope:
                if self.extra_overfit:
                    W_fc1 = variable_with_weight_decay("wfc1", shape=[np.prod(spatial_feat_size)
                                                                       * self.l2depth, self.fully], stddev=.1, wd=0.004)
                else:
                    W_fc1 = weight_variable([np.prod(spatial_feat_size) * self.l2depth, self.fully], "wfc1", self.seed)
                b_fc1 = bias_variable([self.fully], "wbfc1")
                h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(spatial_feat_size) * self.l2depth])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name=scope.name)
                activation_summary(h_fc1,TOWER_NAME)
        else:
            self.keep_prob = tf.placeholder("float",name='keep_prob')
            x_image = tf.reshape(xx, [-1, self.mea_dim[0], self.mea_dim[1], self.inputs, 1])
            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([self.c1size[0], self.c1size[1], self.ctsize, 1, self.l1depth], "wconv1", self.seed)
                b_conv1 = bias_variable([self.l1depth], "wb1")
                h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1,name=scope.name)
                activation_summary(h_conv1,TOWER_NAME)
            if self.extra_overfit:
                h_conv1_drop = tf.nn.dropout(h_conv1, self.keep_prob)
                h_pool1 = max_pool_3d(h_conv1_drop)
            else:
                h_pool1 = max_pool_3d(h_conv1)
            spatial_feat_size = np.ceil(np.array(self.mea_dim)/2.)
            temp_feat_size = np.ceil(self.inputs/2.)
            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([self.c2size[0], self.c2size[1], self.ctsize, self.l1depth, self.l2depth], "wconv2", self.seed)
                b_conv2 = bias_variable([self.l2depth], "wb2")
                h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2,name=scope.name)
                activation_summary(h_conv2,TOWER_NAME)
            if self.extra_overfit:
                h_conv2_drop = tf.nn.dropout(h_conv2, self.keep_prob)
                h_pool2 = max_pool_3d(h_conv2_drop)
            else:
                h_pool2 = max_pool_3d(h_conv2)
            spatial_feat_size = np.array(np.ceil(spatial_feat_size/2.),dtype=int)
            temp_feat_size = int(np.ceil(temp_feat_size / 2.))
            with tf.variable_scope('local3') as scope:
                if self.extra_overfit:
                    W_fc1 = variable_with_weight_decay("wfc1", shape=[
                        np.prod(spatial_feat_size) * temp_feat_size * self.l2depth, self.fully], stddev=.1, wd=0.004)
                else:
                    W_fc1 = weight_variable([np.prod(spatial_feat_size) * temp_feat_size * self.l2depth, self.fully], "wfc1", self.seed)
                b_fc1 = bias_variable([self.fully], "wbfc1")
                h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(spatial_feat_size) * temp_feat_size * self.l2depth])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name=scope.name)
                activation_summary(h_fc1,TOWER_NAME)

        with tf.variable_scope('output') as scope:
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            W_fc2 = weight_variable([self.fully, self.num_categories], "wfc2", self.seed)
            b_fc2 = bias_variable([self.num_categories], "wbfc2")

            # Use simple linear comnination for regrassion (no softmax)
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            activation_summary(y_conv,TOWER_NAME)

        return y_conv

    def loss(self,logits,labels):
        """Add L2 loss to all the trainable variables.
        Additionally add a summary for "Loss" and "Loss/avg".

        Parameters:
        -----------
        logits : TensorFlow tensor
            Logits from self.inference().
        labels : TensorFlow Tensor
            Labels from self.inputs_fct() of shape [batch_size,label_size]

        Returns:
        --------
        Loss : TensorFlow Tensor
        """
        # Calculate the average cross entropy loss across the batch.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


    def training(self,validation_run=None):
        """ Train the network
        
        Parameters:
        -----------
        validation_run : int (optional, default None)
            Number of validation run. If ``None``, it's treaten like zero, 
            but no specification is added to training directory.
        """

        if validation_run is not None:
            tmp_train_dir = join(self.tmp_data_dir,'train_'+str(validation_run))
        else:
            tmp_train_dir = join(self.tmp_data_dir,'train')
            validation_run = 0

        global_step = tf.Variable(0, trainable=False)

        # get data for training
        train_features,train_classes = self.inputs_fct(tmp_train_dir, self.batch_size)
        # prediction
        logits = self.inference(train_features)
        total_loss = self.loss(logits,train_classes)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(train_classes, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('training_accuracy',accuracy)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        # Merge all the summaries
        merged = tf.summary.merge_all()
        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        sess=tf.Session()

        # summary writers for training
        train_writer = tf.summary.FileWriter(join(self.model_path,'train','run%d' % validation_run),sess.graph)

        sess.run(init)

        # start a train coordinator for the input queues
        coord = tf.train.Coordinator()
        # Start the queue runners.
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        ############
        # TRAINING #
        ############
        try:
            t_start = time.time()
            for epoch in range(self.training_epochs):
                if coord.should_stop():
                    break
                sess.run(train_step,feed_dict={self.keep_prob: self.dropout_rate})
                # Display logs per epoch step
                if (epoch+1) % self.display_step == 0:
                    ce,train_accuracy,summary = sess.run([total_loss,accuracy,\
                                    merged],feed_dict={self.keep_prob: 1.0})
                    print("Step:", '%04d' % (epoch + 1), "training accuracy=",\
                          "{:.9f}".format(train_accuracy))
                    print('Elapsed time: ', time.time() - t_start)
                    train_writer.add_summary(summary,epoch)
                if epoch+1 == self.training_epochs:
                    ce, train_accuracy, summary = sess.run([total_loss, accuracy, merged],
                                                           feed_dict={self.keep_prob: 1.0})
                    print("Step:", '%04d' % (epoch + 1), "training accuracy=",\
                          "{:.9f}".format(train_accuracy))
                    self.acc_tr = train_accuracy
                    print('Elapsed time: ', time.time() - t_start)
                    # Save the model checkpoint periodically.
                    if self.save:
                        checkpoint_path = join(self.model_path, 'train','run%d'\
                                               % validation_run ,self.model_name +'.ckpt')
                        saver.save(sess, checkpoint_path ,global_step=epoch)
                        print("Model saved in file: %s" % checkpoint_path)

        except Exception, e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # And wait for them to actually do it.
            coord.join(threads)

        print("Optimization Finished!")
        training_cost = sess.run(total_loss,feed_dict={self.keep_prob: 1.0})

        tf.reset_default_graph()
        sess.close()

    def evaluate(self,validation_run=None):
        """ Evaluate the network

        Parameters:
        -----------
        validation_run : int (optional, default None)
            Number of validation run. If ``None``, it's treaten like zero, 
            but no specification is added to validation directory.
        """


        if validation_run is not None:
            curr_eval_dir = join(self.val_data_dir, 'eval_' + str(validation_run))
        else:
            curr_eval_dir = self.val_data_dir
            validation_run = 0
        checkpoint_path = join(self.model_path, 'train', 'run%d' % validation_run, self.model_name + '.ckpt')

        # get data for testing
        if self.classification == 'binary':
            test_spikes, test_features, test_loc, test_rot, test_cat, test_mcat = \
                load_validation_data(curr_eval_dir,load_mcat=True)
        else:
            test_spikes, test_features, test_loc, test_rot, test_cat = load_validation_data(curr_eval_dir)
        test_features_tf = tf.constant(test_features, dtype=np.float32)
        test_cat_tf = tf.one_hot(test_cat, depth=self.num_categories, dtype=np.int64)
        # prediction
        test_pred = self.inference(test_features_tf)
        true = tf.argmax(test_cat_tf, 1)
        guessed = tf.argmax(test_pred, 1)
        correct_prediction = tf.equal(guessed, true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy',accuracy)

        saver = tf.train.Saver()

        # cost:  Optimize L2 norm
        test_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=test_pred,
                                                                                    labels=test_cat_tf))
        tf.summary.scalar('cross_entropy',test_cross_entropy)

        # Merge all the summaries
        merged_sum = tf.summary.merge_all()

        eval_writer = tf.summary.FileWriter(join(self.model_path,'eval','run%d' % validation_run))

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(join(self.model_path,'train','run%d' % validation_run ))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))


                acc, guess, summary = sess.run([accuracy,guessed,merged_sum],feed_dict={self.keep_prob: 1.0})
                print("Validation accuracy=", "{:.9f}".format(acc))

                self.accuracies.append(acc)
                self.guessed.append(guess)
                self.loc.append(test_loc)
                if self.classification == 'binary':
                    self.cat.append(test_mcat)
                else:
                    self.cat.append(test_cat)
                self.rot.append(test_rot)

                eval_writer.add_summary(summary,global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                # And wait for them to actually do it.
                coord.join(threads)

        tf.reset_default_graph()


    def save_meta_model(self):
        """ Save model data and results. This function creates
            `model_info.yaml` and `results.pkl`.

        """
        # Save meta_info yaml
        print('Saving: ', self.model_path)
        with open(join(self.model_path, 'model_info.yaml'), 'w') as f:
            if self.accuracies.size == 1:
                # not k-fold or k-fold-model
                acc = float(self.accuracies)
                acc_std = 0.
            else:
                # k-fold or k-fold-model
                acc = float(np.mean(self.accuracies))
                acc_std = float(np.std(self.accuracies))
            # ipdb.set_trace()
            general = {'classification': self.classification, 'rotation': self.rotation_type,
                       'n_points': self.n_points, 'pitch': self.pitch, 'electrode name': str(self.electrode_name),
                       'MEA dimension': self.mea_dim, 'spike file': self.spike_file_name,
                       'noise level': self.noise_level, 'time [s]': self.processing_time,
                       'tensorflow': tf.__version__ ,'seed': self.seed}
            cnn = {'learning rate': self.learning_rate, 'size': self.size,
                   'nepochs': self.training_epochs, 'batch size': self.batch_size,
                   'dropout rate': self.dropout_rate, 'inputs': self.inputs, 'c1size': self.c1size,
                   'c2size': self.c2size, 'ctsize': self.ctsize, 'l1depth': self.l1depth, 'l2depth': self.l2depth,
                   'fully': self.fully, 'outputs': self.out}
            validation = {'validation type': self.val_type, 'training size': self.num_train_feat,
                          'validation size': self.num_val_feat, 'validation_runs': self.validation_runs}
            accuracy = {'val_accuracy': acc, 'train_accuracy': float(self.acc_tr)}

            if self.val_type == 'hold-model-out':
                validation.update({'model out': self.model_out})
            elif self.val_type == 'holdout':
                validation.update({'validation percent': self.val_percent})
            elif self.val_type == 'provided-datasets':
                validation.update({'training models': str(list(self.train_cell_names)),
                                   'validation models': str(list(self.val_cell_names))})

            features = {'feature type': self.feat_type,
                        'amp threshold': self.threshold_detect,
                        'dt': self.dt}
            if self.feat_type == '3d':
                features.update({'downsampling factor': self.downsampling_factor})

            # create dictionary for yaml file
            data_yaml = {'General': general,
                         'CNN': cnn,
                         'Validation': validation,
                         'Features': features,
                         'Performance': accuracy
                         }
            yaml.dump(data_yaml, f, default_flow_style=False)

        # create results pkl
        if self.validation_runs > 1:
            for v in range(self.validation_runs):
                n_obs = len(self.guessed[v])

                feat_vec = [self.feat_type] * n_obs
                class_vec = [self.classification] * n_obs
                rot_vec = [self.rotation_type] * n_obs
                valtype_vec = [self.val_type] * n_obs
                elecname_vec = [self.electrode_name] * n_obs
                px_vec = [self.n_points] * n_obs
                mea_ydim_vec = [self.mea_dim[0]] * n_obs
                mea_zdim_vec = [self.mea_dim[1]] * n_obs
                pitch_ydim_vec = [self.pitch[0]] * n_obs
                pitch_zdim_vec = [self.pitch[1]] * n_obs
                size_vec = [self.size] * n_obs
                val_run_vec = [v+1] * n_obs

                cat = [self.cell_dict[cc] for cc in self.cat[v]]
                bin_cat = get_binary_cat(cat,self.exc_categories,self.inh_categories)
                pred_cat = [self.pred_dict[cc] for cc in self.guessed[v]]
                loc = self.loc[v]
                rot = self.rot[v]

                d_obs = {'cell_type': cat, 'binary cat': bin_cat, 'predicted_type': pred_cat,
                         'x': loc[:, 0], 'y': loc[:, 1], 'z': loc[:, 2], 'val_run': val_run_vec,
                         'rot_x': rot[:,0], 'rot_y': rot[:,2], 'rot_z': rot[:,2],
                         'feat_type': feat_vec, 'class_type': class_vec, 'rotation_type': rot_vec,
                         'validation_type': valtype_vec, 'px': px_vec, 'y_pitch': pitch_ydim_vec,
                         'z_pitch': pitch_zdim_vec,'MEA y dimension': mea_ydim_vec, 'MEA z dimension': mea_zdim_vec,
                         'elec_name': elecname_vec, 'size': size_vec,'id': np.arange(n_obs)}

                results_dir = join(self.model_path, 'results')
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                # df_obs.to_csv(join(results_dir, 'results.csv'))
                filen = open(join(results_dir, 'results.pkl'), 'wb')
                pickle.dump(d_obs, filen, protocol=2)
                filen.close()
        else:
            n_obs = len(self.guessed)

            feat_vec = [self.feat_type] * n_obs
            class_vec = [self.classification] * n_obs
            rot_vec = [self.rotation_type] * n_obs
            valtype_vec = [self.val_type] * n_obs
            elecname_vec = [self.electrode_name] * n_obs
            px_vec = [self.n_points] * n_obs
            mea_ydim_vec = [self.mea_dim[0]] * n_obs
            mea_zdim_vec = [self.mea_dim[1]] * n_obs
            pitch_ydim_vec = [self.pitch[0]] * n_obs
            pitch_zdim_vec = [self.pitch[1]] * n_obs
            size_vec = [self.size] * n_obs
            val_run_vec = [0] * n_obs

            cat = [self.cell_dict[cc] for cc in self.cat]
            bin_cat = get_binary_cat(cat,self.exc_categories,self.inh_categories)
            pred_cat = [self.pred_dict[cc] for cc in self.guessed]
            loc = self.loc
            rot = self.rot

            d_obs = {'cell_type': cat, 'binary cat': bin_cat, 'predicted_type': pred_cat,
                     'x': loc[:, 0], 'y': loc[:, 1], 'z': loc[:, 2], 'val_run': val_run_vec,
                     'rot_x': rot[:,0], 'rot_y': rot[:,2], 'rot_z': rot[:,2],
                     'feat_type': feat_vec, 'class_type': class_vec, 'rotation_type': rot_vec,
                     'validation_type': valtype_vec, 'px': px_vec, 'y_pitch': pitch_ydim_vec,
                     'z_pitch': pitch_zdim_vec,'MEA y dimension': mea_ydim_vec, 'MEA z dimension': mea_zdim_vec,
                     'elec_name': elecname_vec, 'size': size_vec,'id': np.arange(n_obs)}

            results_dir = join(self.model_path, 'results')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            # df_obs.to_csv(join(results_dir, 'results.csv'))
            filen = open(join(results_dir, 'results.pkl'), 'wb')
            pickle.dump(d_obs, filen, protocol=2)
            filen.close()


    def get_binary_cat_idx(self, categories):
        """ Get binary category index
        
        Parameters:
        -----------
        categories : list
            List of cell categories.
        """
        binary_cat = []
        for i, cat in enumerate(categories):
            if cat in self.exc_categories:
                binary_cat.append(int(np.argwhere(np.array(self.binary_cat)=='EXCIT')))
            elif cat in self.inh_categories:
                binary_cat.append(int(np.argwhere(np.array(self.binary_cat)=='INHIB')))

        return np.array(binary_cat)

    def get_mtype_cat_idx(self, categories, classes):
        """ Get m-type category index
        
        Parameters:
        -----------
        categories : list
            List of cell categories to get an index for.
        classes : list
            List of m-type classes.
        """
        m_cat = [int(np.argwhere(np.array(classes)==cat)) for cat in categories]
        return np.array(m_cat)

    def remondis(self):
        ''' Cleaning up. 
            Remove temporary directories.
        '''
        if self.tmp_data_dir.split('/')[-1].startswith('tmp'):
            shutil.rmtree(self.tmp_data_dir)
        else:
            raise UserWarning(self.tmp_data_dir+' seems not to be a temporary directory. Did not remove it!')


if __name__ == '__main__':
    '''
        COMMAND-LINE 
        -f filename
        -feat feature type
        -modelout number of model to hold out
        -val validation
        -cl class type
        -cn cellnames
        -n number of training epochs
        -s size
        -tcn training cell names
        -vcn validation cell names
        -seed seed
    '''

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        spike_folder = sys.argv[pos + 1]
    elif len(sys.argv) != 1:
        raise Exception('Provide spikes folder with argument -f')
    if '-feat' in sys.argv:
        pos = sys.argv.index('-feat')
        feat_type = sys.argv[pos + 1]
    else:
        feat_type = 'FW'
    if '-val' in sys.argv:
        pos = sys.argv.index('-val')
        val_type = sys.argv[pos + 1]
    else:
        val_type = 'holdout'
    if '-modelout' in sys.argv:
        pos = sys.argv.index('-modelout')
        model_out = int(sys.argv[pos + 1])
    else:
        model_out = int(5)
    if '-cl' in sys.argv:
        pos = sys.argv.index('-cl')
        class_type = sys.argv[pos + 1]
    else:
        class_type = 'binary'
    if '-n' in sys.argv:
        pos = sys.argv.index('-n')
        nsteps = int(sys.argv[pos+1])
    else:
        nsteps = 2000
    if '-cn' in sys.argv:
        pos = sys.argv.index('-cn')
        cell_names = sys.argv[pos + 1]
    else:
        cell_names = 'all'
    if '-tcn' in sys.argv:
        pos = sys.argv.index('-tcn')
        train_cell_names = sys.argv[pos+1]  # all - 'filename'
    else:
        train_cell_names = None
    if '-vcn' in sys.argv:
        pos = sys.argv.index('-vcn')
        val_cell_names = sys.argv[pos+1]  # all - 'filename'
    else:
        val_cell_names = None
    if '-s' in sys.argv:
        pos = sys.argv.index('-s')
        size = sys.argv[pos+1]  # 'xs' - 's' - 'm' - 'l' - 'xl'
    else:
        size = 'l'
    if '-seed' in sys.argv:
        pos = sys.argv.index('-seed')
        seed = int(sys.argv[pos+1])  
    else:
        seed = int(2308)
    if len(sys.argv) == 1:
        print('Arguments: \n   -f full-path\n'\
              '-feat feature type: AW - FW - AFW - 3d\n' \
              '-val validation: holdout,k-fold,hold-model-out,k-fold-model\n'\
              '-cn cellnames: all - filename\n '\
              '-tcn train cellnames file\n '\
              '-vcn validation cellnames file\n' \
              '-cl classification: binary - m-type\n' \
              '-s  size: xs - s - m - l - xl\n'\
              '-modelout model to hold out'\
              '-seed random seed (integer)')
    else:
        cv = SpikeConvNet(train=True, save=True, spike_folder=spike_folder,\
                          feat_type=feat_type, class_type=class_type,\
                          val_type=val_type, cellnames=cell_names, size=size,\
                          model_out=model_out, nsteps=nsteps,\
                          train_cell_names=train_cell_names,\
                          val_cell_names=val_cell_names,seed=seed)
