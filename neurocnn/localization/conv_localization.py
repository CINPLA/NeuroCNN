'''

Deep net to localize soma position based on NA trough and Rip peak

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

from ..tools import *
from ..tftools import *


class LocalizationNetwork:
    """ Convolutional Neural Network class for neuron classification
    based on extracellular action potentials (EAPs) measured with 
    Multi-Electrode Arrays (MEAs)

    Parameters:
    -----------
    save, bool (optional, default True)
        Whether to save the results
    train, bool (optional, default True)
        Whether to train the network
    noise_level, float (optional, default None)
        Standard deviation for additive Gaussian signal noise.
        If ``None``, no noise is added.
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
    n_spikes, int (optional, default None)
        Number of spikes taken into account. If ``None``, all available spikes
        are taken into account (in a balanced manner).
    val_percent, float (optional, default=10.)
        Percentage of data to be left out for validation in combination with
        val_type=``holdout``.
    nsteps, int (optional, default=2000)
        Number of training epochs,
    feat_type, str (optional, defauls 'NaRep')
        String which specifies features of EAPs to be used by the CNN. Possible 
        values are 'Na','NaRep','Rep','3d'.
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
    seed, int (optional, default=2308)
        Random seed for TensorFlow variable initialization. 
    """
    def __init__(self, save=True, train=True, noise_level=None, keep_tmp_data=False,
                 spike_folder=None, cellnames='all', train_cell_names=None, val_cell_names=None,
                 n_spikes=None, val_percent=10., nsteps=2000,
                 feat_type='NaRep', val_type='holdout', kfolds=5, size='l',
                 model_out=5, seed=2308):
        """ Initialization
        """
        if not noise_level is None:
            self.noise_level = noise_level
        else:
            self.noise_level = 0  # uV
        self.train = train
        self.save = save
        self.seed = seed

        # Na - NaRep - Rep - 3d
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
                                     os.listdir(join(data_dir, 'spikes', os.listdir(join(data_dir, 'spikes'))[-1]))[
                                         0])
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

        self.all_etypes = ['cADpyr', 'cAC', 'bAC', 'cNAC', 'bNAC', 'dNAC', 'cSTUT', 'bSTUT', 'dSTUT', 'cIR', 'bIR']

        self.val_percent = val_percent

        if train_cell_names:
            self.train_cell_names = list(np.loadtxt(join(root_folder, 'localization', train_cell_names), dtype=str))
            print('cells from file used for training: ', self.train_cell_names)
            if val_cell_names:
                self.val_cell_names = list(np.loadtxt(join(root_folder, 'localization', val_cell_names), dtype=str))
                print('cells from file used for validationm: ', self.val_cell_names)
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
                self.cell_names = list(np.loadtxt(join(root_folder, 'localization', cellnames), dtype=str))
                np.random.shuffle(self.cell_names)
                print('cells from file used for training: ', self.cell_names)

        self.dt = 2 ** -5

        # Setup Network parameters
        self.learning_rate = 5e-4
        self.training_epochs = nsteps
        self.batch_size = 1000
        self.display_step = 10
        self.dropout_rate = 0.7
        # Network Parameters
        if self.feat_type == 'NaRep':
            self.inputs = 2
            self.ctsize = 0
        elif self.feat_type == 'Na':
            self.inputs = 1
            self.ctsize = 0
        elif self.feat_type == 'Rep':
            self.inputs = 1
            self.ctsize = 0
        elif self.feat_type == '3d':
            self.downsampling_factor = 16 # 8 # 4
            self.ctsize = 4
            # self.ctsize = 2
        self.out = 3  # x, y, z

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
        self.model_name = 'model_' + self.rotation_type + '_'  + \
                         self.feat_type + '_' + self.val_type + '_' + \
                         self.size + '_' + self.spike_file_name + '_' + \
                         time.strftime("%d-%m-%Y:%H:%M")
        loc_data_dir = join(data_dir, 'localization')
        self.model_path = join(loc_data_dir, 'models', self.model_name)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.ndatafiles = 10
        self.datacounter = 0
        self.datafilenames = ['data_batch_%d.csv' % fn for fn in range(self.ndatafiles) ]

        self.cat_in_dset = np.unique([f.split('_')[1] for f in self.cell_names])
        self.cell_dict = {}
        for cc in self.cat_in_dset:
            self.cell_dict.update({int(np.argwhere(np.array(self.all_categories) == cc)): cc})

        self.class_grp = 'all'

        self.val_data_dir = join(self.model_path, 'validation_data')
        if not os.path.isdir(self.val_data_dir):
            os.makedirs(self.val_data_dir)

        ######################
        # data preprocessing #
        ######################
        self.class_categories = set()
        self.pred_categories = set()
        self._data_preprocessing(self.all_categories)

        self.class_categories = list(self.class_categories)
        self.num_categories = len(self.class_categories)

        ############
        # training #
        ############
        if self.train:
            t_start = time.time()

            self.accuracies, self.preds, self.loc, self.rot, self.cat, self.errors, \
            self.errors_dim = [], [], [], [], [], [], []
            self.batch_size = int(self.num_train_feat / 10.)
            if self.validation_runs > 1:
                for v in range(self.validation_runs):
                    print('Training/Validation run: ', v + 1, '/', self.validation_runs)
                    self.training(validation_run=v)
                    self.evaluate(validation_run=v)
            else:
                self.training()
                self.evaluate()

            self.accuracies = np.squeeze(np.array(self.accuracies))
            self.preds = np.squeeze(np.array(self.preds))
            self.loc = np.squeeze(np.array(self.loc))
            self.rot = np.squeeze(np.array(self.rot))
            self.cat = np.squeeze(np.array(self.cat))
            self.errors = np.squeeze(np.array(self.errors))
            self.errors_dim = np.squeeze(np.array(self.errors_dim))

            print("Training done")
            t_end = time.time()
            self.processing_time = t_end - t_start
            print('Training time: ', self.processing_time)

            # final evaluation
            self.final_evaluate()

            if self.save:
                self.save_meta_model()

        # cleaning up
        if not keep_tmp_data:
            self.remondis()


    def _data_preprocessing(self, categories, balanced=True):
        ''' organize training and validation data, extract the features
        put everything into temporary directions, from which an input queue is created in tensorflow
        '''
        if not self.train_cell_names:
            # Get even number of observations per class
            spikelist = [f for f in os.listdir(self.spike_folder) if
                         f.startswith('e_spikes') and any(x in f for x in self.cell_names)]
            cells, occurrences = np.unique(sum([[f.split('_')[4]] * int(f.split('_')[2]) for f in spikelist], []),
                                           return_counts=True)
            occ_dict = dict(zip(cells, occurrences))
            min_occurrence = np.min(occurrences)
            if self.n_spikes is not 'all':
                min_occurrence = self.n_spikes

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
            self.tmp_data_dir = join(data_dir, 'localization', 'tmp' + time.strftime("%d-%m-%Y:%H:%M")+str(np.random.rand()).split('.')[1])
            if not os.path.isdir(self.tmp_data_dir):
                os.makedirs(self.tmp_data_dir)
            else:
                raise IOError('Temporary data directory exists already.')

            self.num_train_feat = 0
            self.num_val_feat = 0


            # do loading and feature extraction for each cell separately
            for idx, cell in enumerate(categories):
                avail_samples = sum([int(f.split('_')[2]) for f in spikelist if cell in f])
                print(idx+1, '/', len(categories), ' :', cell, ' avail: ', avail_samples)
                n_samples = min_occurrence  #int(avail_samples / occ_dict[cell] * min_occurrence)
                cells_to_load = [c for c in self.cell_names if cell in c]
                spikes, loc, rot, cat, etype, morphid, loaded_cat = load_EAP_data(self.spike_folder, \
                                                                                  cells_to_load, categories)
                # subsample min_occurrences
                if balanced:
                    if len(cat) > min_occurrence:
                        rand_per = np.random.permutation(len(cat))
                        idx_cells  = rand_per[:min_occurrence]
                        spikes = spikes[idx_cells]
                        loc = loc[idx_cells]
                        rot = rot[idx_cells]
                        cat = cat[idx_cells]
                        etype = etype[idx_cells]
                        morphid = morphid[idx_cells]

                if len(cat) > 0:

                    cat = self.get_mtype_cat_idx(cat, self.all_categories)
                    # self.cell_dict.update({int(np.unique(cat)): cell})
                    self.class_categories.update(loaded_cat)

                    # # convert etype to integer
                    for i, typ in enumerate(etype):
                        for j, et in enumerate(self.all_etypes):
                            if et in typ:
                                etype[i] = j

                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(spikes.shape[0], spikes.shape[1], spikes.shape[2]))
                        spikes += noise
                    # divide train an validation data
                    self.create_tmp_data(spikes, loc, rot, cat, etype, morphid)

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

            val_spikelist = [f for f in os.listdir(self.spike_folder) if
                               f.startswith('e_spikes') and any(x in f for x in self.val_cell_names)]
            val_cells, val_occurrences = np.unique(
                sum([[f.split('_')[4]] * int(f.split('_')[2]) for f in val_spikelist], []),
                return_counts=True)
            val_occ_dict = dict(zip(val_cells, val_occurrences))
            val_min_occurrence = np.min(val_occurrences)

            if self.n_spikes is not 'all':
                train_min_occurrence = self.n_spikes
                val_min_occurrence = self.n_spikes

            # set up temporary directories
            self.tmp_data_dir = join(data_dir, 'localization',
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
                train_cells_to_load = [c for c in self.train_cell_names if cell in c]
                train_spikes, train_loc, train_rot, train_cat, train_etype, train_morphid, loaded_cat = \
                    load_EAP_data(self.spike_folder, train_cells_to_load, categories)

                # subsample min_occurrences
                if balanced:
                    if len(train_cat) > train_min_occurrence:
                        rand_per = np.random.permutation(len(train_cat))
                        idx_cells = rand_per[:train_min_occurrence]
                        train_spikes = train_spikes[idx_cells]
                        train_loc = train_loc[idx_cells]
                        train_rot = train_rot[idx_cells]
                        train_cat = train_cat[idx_cells]
                        train_etype = train_etype[idx_cells]
                        train_morphid = train_morphid[idx_cells]

                n_train = len(train_cat)

                if len(train_cat) > 0:
                    train_cat = self.get_mtype_cat_idx(train_cat, self.all_categories)
                    self.class_categories.update(loaded_cat)

                    # # convert etype to integer
                    for i, typ in enumerate(train_etype):
                        for j, et in enumerate(self.all_etypes):
                            if et in typ:
                                train_etype[i] = j
                    #
                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(train_spikes.shape[0], train_spikes.shape[1],
                                                       train_spikes.shape[2]))
                        train_spikes += noise
                    # divide train an validation data
                    self.create_tmp_data(train_spikes, train_loc, train_rot, train_cat, train_etype, train_morphid,
                                         dset='train')

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
                val_cells_to_load = [c for c in self.val_cell_names if cell in c]
                val_spikes, val_loc, val_rot, val_cat, val_etype, val_morphid, loaded_cat = \
                    load_EAP_data(self.spike_folder, val_cells_to_load, categories)
                print(idx + 1, '/', len(categories), ' :', cell, ' avail train: ', train_avail_samples, \
                    ' avail val: ', val_avail_samples)

                # subsample min_occurrences
                if balanced:
                    if len(val_cat) > val_min_occurrence:
                        rand_per = np.random.permutation(len(val_cat))
                        idx_cells = rand_per[:val_min_occurrence]
                        val_spikes = val_spikes[idx_cells]
                        val_loc = val_loc[idx_cells]
                        val_rot = val_rot[idx_cells]
                        val_cat = val_cat[idx_cells]
                        val_etype = val_etype[idx_cells]
                        val_morphid = val_morphid[idx_cells]

                n_val = len(val_cat)

                if len(val_cat) > 0:
                    val_cat = self.get_mtype_cat_idx(val_cat, self.all_categories)
                    self.class_categories.update(loaded_cat)

                    # # convert etype to integer
                    for i, typ in enumerate(val_etype):
                        for j, et in enumerate(self.all_etypes):
                            if et in typ:
                                val_etype[i] = j
                    #
                    # Add noise to spikes
                    if self.noise_level > 0:
                        noise = np.random.normal(0, self.noise_level,
                                                 size=(val_spikes.shape[0], val_spikes.shape[1],
                                                       val_spikes.shape[2]))
                        val_spikes += noise
                    # divide train an validation data
                    self.create_tmp_data(val_spikes, val_loc, val_rot, val_cat, val_etype, val_morphid,
                                         dset='validation')



    def return_features(self, spikes):
        ''' extract features from spikes
        '''
        # Create feature set:
        if self.feat_type == 'Na':
            features = get_EAP_features(spikes, ['Na'], \
                                        dt=self.dt, threshold_detect=self.threshold_detect)
            features = np.array(features['na'])
            self.inputs = 1
        elif self.feat_type == 'Rep':
            features = get_EAP_features(spikes, ['Rep'], \
                                        dt=self.dt, threshold_detect=self.threshold_detect)
            features = np.array(features['rep'])
            self.inputs = 1
        elif self.feat_type == 'NaRep':
            features = get_EAP_features(spikes, ['Na', 'Rep'], \
                                        dt=self.dt, threshold_detect=self.threshold_detect)
            features = np.array([features['na'], features['rep']])
            self.inputs = 2  # fwhm and widths
            # swapaxes to move features at the end
            features = features.swapaxes(0, 1)
            features = features.swapaxes(1, 2)
        elif self.feat_type == '3d':
            print('Downsampling spikes...')
            # downsampled_spikes = ss.resample(self.spikes, self.spikes.shape[2] // self.downsampling_factor, axis=2)
            downsampled_spikes = spikes[:, :, ::int(self.downsampling_factor)]
            features = downsampled_spikes
            self.inputs = spikes.shape[2] // self.downsampling_factor
            print('Done')

        return features

    def create_tmp_data(self, spikes, loc, rot, cat, etype, morphid, dset=None):
        ''' devide train and validation data and store it temporary
        '''
        if not dset:
            num_spikes = len(list(cat))

            # shuffle observaitons:
            shuffle_idx = np.random.permutation(num_spikes)
            spikes = spikes[shuffle_idx]
            loc = loc[shuffle_idx]
            rot = rot[shuffle_idx]
            cat = cat[shuffle_idx]
            etype = etype[shuffle_idx]
            morphid = morphid[shuffle_idx]

            if self.val_type == 'holdout':
                self.validation_runs = 1
                tmp_train_dir = join(self.tmp_data_dir, 'train')
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
                validation_feat = self.return_features(validation_spikes)
                validation_loc = loc[-validation_size:]
                validation_rot = rot[-validation_size:]
                validation_cat = cat[-validation_size:]
                validation_etype = etype[-validation_size:]

                self.num_train_feat += len(train_cat)
                self.num_val_feat += len(validation_cat)

                self.dump_learndata(tmp_train_dir, [train_loc, train_rot, train_cat, train_etype, train_feat])
                self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                  validation_rot, validation_cat)

            elif self.val_type == 'k-fold':
                self.validation_runs = self.kfolds
                subsample_size = int(num_spikes / self.kfolds)
                for kk in range(self.kfolds):
                    tmp_train_dir = join(self.tmp_data_dir, 'train_' + str(kk))
                    curr_eval_dir = join(self.val_data_dir, 'eval_' + str(kk))
                    if not os.path.isdir(tmp_train_dir):
                        os.makedirs(tmp_train_dir)
                    if not os.path.isdir(curr_eval_dir):
                        os.makedirs(curr_eval_dir)

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
                    validation_feat = self.return_features(validation_spikes)
                    validation_loc = loc[val_idx]
                    validation_rot = rot[val_idx]
                    validation_cat = cat[val_idx]
                    validation_etype = etype[val_idx]

                    self.dump_learndata(tmp_train_dir, [train_loc, train_rot, train_cat, train_etype, train_feat])
                    self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                      validation_rot, validation_cat)

                self.num_train_feat += len(train_cat)
                self.num_val_feat += len(validation_cat)

            elif self.val_type == 'hold-model-out':
                self.validation_runs = 1
                tmp_train_dir = join(self.tmp_data_dir, 'train')
                curr_eval_dir = self.val_data_dir
                if not os.path.isdir(tmp_train_dir):
                    os.makedirs(tmp_train_dir)
                if not os.path.isdir(curr_eval_dir):
                    os.makedirs(curr_eval_dir)
                if self.num_morph > 1:
                    train_idx = np.where(morphid != self.model_out)
                    val_idx = np.where(morphid == self.model_out)

                    train_spikes = spikes[train_idx]
                    if len(train_idx[0]) != 0:
                        train_feat = self.return_features(train_spikes)
                        train_loc = loc[train_idx]
                        train_rot = rot[train_idx]
                        train_cat = cat[train_idx]
                        train_etype = etype[train_idx]
                        self.num_train_feat += len(train_cat)
                        self.dump_learndata(tmp_train_dir, [train_loc, train_rot, train_cat, train_etype, train_feat])

                    validation_spikes = spikes[val_idx]
                    if len(val_idx[0]) != 0:
                        validation_feat = self.return_features(validation_spikes)
                        validation_loc = loc[val_idx]
                        validation_rot = rot[val_idx]
                        validation_cat = cat[val_idx]
                        validation_etype = etype[val_idx]
                        self.num_val_feat += len(validation_cat)
                        self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                          validation_rot, validation_cat)

                else:
                    raise ValueError(
                        'Cross model validation can be performed only when multiple cell morphologies of same '
                        'me-type are availabel in the dataset.')

            elif self.val_type == 'k-fold-model':
                self.validation_runs = len(self.morph_ids)
                for kk in range(len(self.morph_ids)):
                    tmp_train_dir = join(self.tmp_data_dir, 'train_' + str(kk))
                    curr_eval_dir = join(self.val_data_dir, 'eval_' + str(kk))
                    if not os.path.isdir(tmp_train_dir):
                        os.makedirs(tmp_train_dir)
                    if not os.path.isdir(curr_eval_dir):
                        os.makedirs(curr_eval_dir)

                    train_idx = np.where(morphid != self.morph_ids[kk])
                    val_idx = np.where(morphid == self.morph_ids[kk])

                    train_spikes = spikes[train_idx]
                    if len(train_idx[0]) != 0:
                        train_feat = self.return_features(train_spikes)
                        train_loc = loc[train_idx]
                        train_rot = rot[train_idx]
                        train_cat = cat[train_idx]
                        train_etype = etype[train_idx]
                        if kk == 0:
                            self.num_train_feat += len(train_cat)
                        self.dump_learndata(tmp_train_dir, [train_loc, train_rot, train_cat, train_etype, train_feat])

                    validation_spikes = spikes[val_idx]
                    if len(val_idx[0]) != 0:
                        validation_feat = self.return_features(validation_spikes)
                        validation_loc = loc[val_idx]
                        validation_rot = rot[val_idx]
                        validation_cat = cat[val_idx]
                        validation_etype = etype[val_idx]
                        if kk == 0:
                            self.num_val_feat += len(validation_cat)
                        self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                                          validation_rot, validation_cat)
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
            curr_eval_dir = self.val_data_dir
            if not os.path.isdir(curr_eval_dir):
                os.makedirs(curr_eval_dir)

            validation_spikes = spikes

            validation_feat = self.return_features(validation_spikes)
            validation_loc = loc
            validation_rot = rot
            validation_cat = cat
            validation_etype = etype
            self.num_val_feat += len(validation_cat)

            self.dump_valdata(curr_eval_dir, validation_spikes, validation_feat, validation_loc,
                              validation_rot, validation_cat)

    def dump_learndata(self,filedir,data):
        ''' dump the data to files in filedir'''

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

    def dump_valdata(self, val_dir, spikes, feat, loc, rot, cat):
        ''' dump validation data in npy files
        '''
        if not os.path.isfile(join(val_dir, 'val_spikes.npy')):
            np.save(join(val_dir, 'val_spikes'), spikes)
            np.save(join(val_dir, 'val_feat'), feat)
            np.save(join(val_dir, 'val_loc'), loc)
            np.save(join(val_dir, 'val_rot'), rot)
            np.save(join(val_dir, 'val_cat'), cat)
        else:
            np.save(join(val_dir, 'val_spikes'), np.vstack((np.load(join(val_dir, 'val_spikes.npy')), spikes)))
            np.save(join(val_dir, 'val_feat'), np.vstack((np.load(join(val_dir, 'val_feat.npy')), feat)))
            np.save(join(val_dir, 'val_loc'), np.vstack((np.load(join(val_dir, 'val_loc.npy')), loc)))
            np.save(join(val_dir, 'val_rot'), np.vstack((np.load(join(val_dir, 'val_rot.npy')), rot)))
            np.save(join(val_dir, 'val_cat'), np.concatenate((np.load(join(val_dir, 'val_cat.npy')), cat)))

    def read_single_example(self, filename_queue):
        ''' get a single example from filename queue
        '''

        class NclassRecord(object):
            pass

        result = NclassRecord()

        # data specific parameters
        label_sizes = [3, 3, 1, 1]  # loc,rot,cat,etype
        feature_size = self.n_elec * self.inputs
        record_size = np.sum(label_sizes) + feature_size

        # Create Reader
        reader = tf.TextLineReader()
        result.key, value = reader.read(filename_queue)

        # data preprocessing
        record_defaults = list(np.zeros((record_size, 1), dtype='float32'))
        record_size = tf.decode_csv(value, record_defaults=record_defaults)

        result.loclabel = tf.cast(tf.slice(record_size, [0], [3]), tf.float32)
        result.rotlabel = tf.cast(tf.slice(record_size, [3], [3]), tf.float32)
        result.mlabel = tf.cast(tf.slice(record_size, [6], [1]), tf.int64)
        result.elabel = tf.cast(tf.slice(record_size, [7], [1]), tf.int64)
        # label as one hot

        result.mlabel = tf.SparseTensor([result.mlabel], [1.], dense_shape=[self.num_categories])
        result.mlabel = tf.sparse_tensor_to_dense(result.mlabel)

        result.features = tf.reshape(tf.slice(record_size, [np.sum(label_sizes)], [feature_size]),
                                     [self.n_elec, self.inputs])
        return result

    def inputs_fct(self, ddir, batch):
        ''' Returns
        feature_batch
        label_batch
        '''
        # filenames
        filenames = [join(ddir, f) for f in os.listdir(ddir) if f.endswith('.csv')]
        # filename queue
        filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

        # read one example from filename queue
        read_input = self.read_single_example(filename_queue)
        features = read_input.features
        loclabel = read_input.loclabel

        # generate the batch
        feature_batch, loclabel_batch = tf.train.shuffle_batch([features, loclabel],
                                                               batch_size=batch,
                                                               num_threads=2,
                                                               capacity=65000,
                                                               min_after_dequeue=30000)
        return feature_batch, loclabel_batch

    def inference(self, xx):
        ''' infer network prediction
        Parameters:
        -----------
        xx: tensor, graph input
        Return:
        -------
        pred: tensor, prediction
        '''
        # if self.feat_type != '3d':
        if self.feat_type != '3d':
            x_image = tf.reshape(xx, [-1, self.mea_dim[0], self.mea_dim[1], self.inputs])
            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([self.c1size[0], self.c1size[1], self.inputs, self.l1depth], "wconv1",
                                          self.seed)
                b_conv1 = bias_variable([self.l1depth], "wb1")
                h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
                activation_summary(h_conv1,TOWER_NAME)
            h_pool1 = max_pool_2d(h_conv1)
            spatial_feat_size = np.ceil(np.array(self.mea_dim) / 2.)
            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([self.c2size[0], self.c2size[1], self.l1depth, self.l2depth], "wconv2",
                                          self.seed)
                b_conv2 = bias_variable([self.l2depth], "wb2")
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name=scope.name)
                activation_summary(h_conv2,TOWER_NAME)
            h_pool2 = max_pool_2d(h_conv2)
            spatial_feat_size = np.array(np.ceil(spatial_feat_size / 2.), dtype=int)
            with tf.variable_scope('local3') as scope:
                W_fc1 = weight_variable([np.prod(spatial_feat_size) * self.l2depth, self.fully], "wfc1", self.seed)
                b_fc1 = bias_variable([self.fully], "wbfc1")
                h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(spatial_feat_size) * self.l2depth])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name=scope.name)
                activation_summary(h_fc1,TOWER_NAME)
        else:
            x_image = tf.reshape(xx, [-1, self.mea_dim[0], self.mea_dim[1], self.inputs, 1])
            with tf.variable_scope('conv1') as scope:
                W_conv1 = weight_variable([self.c1size[0], self.c1size[1], self.ctsize, 1, self.l1depth], "wconv1",
                                          self.seed)
                b_conv1 = bias_variable([self.l1depth], "wb1")
                h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1, name=scope.name)
                activation_summary(h_conv1,TOWER_NAME)
            h_pool1 = max_pool_3d(h_conv1)
            spatial_feat_size = np.ceil(np.array(self.mea_dim) / 2.)
            temp_feat_size = np.ceil(self.inputs / 2.)
            with tf.variable_scope('conv2') as scope:
                W_conv2 = weight_variable([self.c2size[0], self.c2size[1], self.ctsize, self.l1depth, self.l2depth],
                                          "wconv2", self.seed)
                b_conv2 = bias_variable([self.l2depth], "wb2")
                h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2, name=scope.name)
                activation_summary(h_conv2,TOWER_NAME)
            h_pool2 = max_pool_3d(h_conv2)
            spatial_feat_size = np.array(np.ceil(spatial_feat_size / 2.), dtype=int)
            temp_feat_size = int(np.ceil(temp_feat_size / 2.))
            with tf.variable_scope('local3') as scope:
                W_fc1 = weight_variable([np.prod(spatial_feat_size) * temp_feat_size * self.l2depth, self.fully],
                                        "wfc1", self.seed)
                b_fc1 = bias_variable([self.fully], "wbfc1")
                h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(spatial_feat_size) * temp_feat_size * self.l2depth])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name=scope.name)
                activation_summary(h_fc1,TOWER_NAME)

        with tf.variable_scope('output') as scope:
            self.keep_prob = tf.placeholder("float", name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            W_fc2 = weight_variable([self.fully, self.out], "wfc2", self.seed)
            b_fc2 = bias_variable([self.out], "wbfc2")

            # Use simple linear comnination for regrassion (no softmax)
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            activation_summary(y_conv,TOWER_NAME)

        return y_conv

    def training(self, validation_run=None):
        ''' train the network
        '''
        if validation_run is not None:
            tmp_train_dir = join(self.tmp_data_dir, 'train_' + str(validation_run))
        else:
            tmp_train_dir = join(self.tmp_data_dir, 'train')
            validation_run = 0

        global_step = tf.Variable(0, trainable=False)

        # get data for training
        train_features, train_loc = self.inputs_fct(tmp_train_dir, self.batch_size)
        # prediction
        train_pred = self.inference(train_features)
        # cost:  Optimize L2 norm
        loss_fun = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_loc, train_pred)),
                                                        reduction_indices=1)))
        tf.summary.scalar('loss_fun', loss_fun)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_fun)

        accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(train_loc, train_pred)),
                                                        reduction_indices=1)))
        tf.summary.scalar('training_accuracy', accuracy)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        # Merge all the summaries
        merged = tf.summary.merge_all()
        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        sess = tf.Session()

        # summary writers for training
        train_writer = tf.summary.FileWriter(join(self.model_path, 'train', 'run%d' % validation_run), sess.graph)

        sess.run(init)

        # start a train coordinator for the input queues
        coord = tf.train.Coordinator()
        # Start the queue runners.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ############
        # TRAINING #
        ############
        try:
            t_start = time.time()
            for epoch in range(self.training_epochs):
                if coord.should_stop():
                    break
                sess.run(train_step, feed_dict={self.keep_prob: self.dropout_rate})

                # Display logs per epoch step
                if (epoch + 1) % self.display_step == 0:
                    train_accuracy, summary = sess.run([accuracy, merged],
                                                           feed_dict={self.keep_prob: 1.0})
                    print("Step:", '%04d' % (epoch + 1), "training accuracy=", "{:.9f}".format(train_accuracy))
                    print('Elapsed time: ', time.time() - t_start)
                    train_writer.add_summary(summary, epoch)
                if epoch + 1 == self.training_epochs:
                    train_accuracy, summary = sess.run([accuracy, merged],
                                                       feed_dict={self.keep_prob: 1.0})
                    print("Final Step:", '%04d' % (epoch + 1), "training accuracy=", "{:.9f}".format(train_accuracy))
                    self.acc_tr = train_accuracy
                    print('Elapsed time: ', time.time() - t_start)
                    # Save the model checkpoint periodically.
                    if self.save:
                        checkpoint_path = join(self.model_path, 'train', 'run%d' % validation_run,
                                               self.model_name + '.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)
                        print("Model saved in file: %s" % checkpoint_path)

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # And wait for them to actually do it.
            coord.join(threads)

        print("Optimization Finished!")
        training_loss = sess.run(loss_fun, feed_dict={self.keep_prob: 1.0})
        print("Training loss function (validation run ", validation_run, ") =", training_loss, '\n')

        tf.reset_default_graph()
        sess.close()

    def evaluate(self, validation_run=None):
        ''' evaluate the network
        '''
        if validation_run is not None:
            curr_eval_dir = join(self.val_data_dir, 'eval_' + str(validation_run))
        else:
            curr_eval_dir = self.val_data_dir
            validation_run = 0
        checkpoint_path = join(self.model_path, 'train', 'run%d' % validation_run, self.model_name + '.ckpt')

        # get data for testing
        test_spikes, test_features, test_loc, test_rot, test_cat = load_validation_data(curr_eval_dir)
        test_features_tf = tf.constant(test_features, dtype=np.float32)
        test_loc_tf = tf.constant(test_loc, dtype=np.float32)
        # prediction
        test_pred = self.inference(test_features_tf)
        accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(test_loc_tf, test_pred)),
                                                        reduction_indices=1)))
        saver = tf.train.Saver()

        # cost:  Optimize L2 norm
        test_loss_fun = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(test_loc_tf, test_pred)),
                                                        reduction_indices=1)))
        tf.summary.scalar('loss_fun', test_loss_fun)

        # Merge all the summaries
        merged_sum = tf.summary.merge_all()

        eval_writer = tf.summary.FileWriter(join(self.model_path, 'eval', 'run%d' % validation_run))

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(join(self.model_path, 'train', 'run%d' % validation_run))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
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

                acc, pred, summary = sess.run([accuracy, test_pred, merged_sum],
                                              feed_dict={self.keep_prob: 1.0})
                print("Validation accuracy=", "{:.9f}".format(acc))

                err = np.array([np.linalg.norm(pred[e, :] - test_loc[e, :]) for e in range(pred.shape[0])])
                err_x = np.array([np.linalg.norm(pred[e, 0] - test_loc[e, 0]) for e in range(pred.shape[0])])
                err_y = np.array([np.linalg.norm(pred[e, 1] - test_loc[e, 1]) for e in range(pred.shape[0])])
                err_z = np.array([np.linalg.norm(pred[e, 2] - test_loc[e, 2]) for e in range(pred.shape[0])])
                err_dim = [err_x, err_y, err_z]

                self.accuracies.append(acc)
                self.preds.append(pred)
                self.loc.append(test_loc)
                self.rot.append(test_rot)
                self.cat.append(test_cat)
                self.errors.append(err)
                self.errors_dim.append(err_dim)


            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                # And wait for them to actually do it.
                coord.join(threads)

        tf.reset_default_graph()

    def final_evaluate(self):
        ''' some final evaluation after all validation runs can be implemented here
        '''
        pass

    def save_meta_model(self):
        ''' save meta data in old fashioned format'''
        # Save meta_info yaml
        print('Saving: ', self.model_path)
        with open(join(self.model_path, 'model_info.yaml'), 'w') as f:
            if self.accuracies.size == 1:
                # not k-fold or k-fold-model
                acc = float(self.accuracies)
                acc_std = 0.
                err = float(np.mean(self.errors))
                err_x = float(np.mean(self.errors_dim[0]))
                err_y = float(np.mean(self.errors_dim[1]))
                err_z = float(np.mean(self.errors_dim[2]))
                err_sd = float(np.std(self.errors))
                err_x_sd = float(np.std(self.errors_dim[0]))
                err_y_sd = float(np.std(self.errors_dim[1]))
                err_z_sd = float(np.std(self.errors_dim[2]))
            else:
                # k-fold or k-fold-model
                acc = float(np.mean(self.accuracies))
                acc_std = float(np.std(self.accuracies))
                err = float(np.mean([np.mean(e) for e in self.errors]))
                err_x = float(np.mean([np.mean(self.errors_dim[e, 0]) for e in range(self.validation_runs)]))
                err_y = float(np.mean([np.mean(self.errors_dim[e, 1]) for e in range(self.validation_runs)]))
                err_z = float(np.mean([np.mean(self.errors_dim[e, 2]) for e in range(self.validation_runs)]))
                err_sd = float(np.mean([np.std(e) for e in self.errors]))
                err_x_sd = float(np.mean([np.std(self.errors_dim[e, 0]) for e in range(self.validation_runs)]))
                err_y_sd = float(np.mean([np.std(self.errors_dim[e, 1]) for e in range(self.validation_runs)]))
                err_z_sd = float(np.mean([np.std(self.errors_dim[e, 2]) for e in range(self.validation_runs)]))
            # ipdb.set_trace()
            general = {'rotation': self.rotation_type, 'cell specific': self.class_grp,
                       'n_points': self.n_points, 'pitch' : self.pitch, 'electrode name': self.electrode_name,
                       'MEA dimension': self.mea_dim, 'spike file': self.spike_file_name,
                       'noise level': self.noise_level, 'time [s]': self.processing_time,
                       'tensorflow': tf.__version__, 'seed': self.seed}
            cnn = {'learning rate': self.learning_rate, 'size': self.size,
                   'nepochs': self.training_epochs, 'batch size': self.batch_size,
                   'dropout rate': self.dropout_rate, 'inputs': self.inputs, 'c1size': self.c1size,
                   'c2size': self.c2size, 'ctsize': self.ctsize, 'l1depth': self.l1depth, 'l2depth': self.l2depth,
                   'fully': self.fully, 'outputs': self.out}
            validation = {'validation type': self.val_type, 'training size': self.num_train_feat,
                          'validation size': self.num_val_feat, 'validation_runs': self.validation_runs}

            accuracy = {'val_accuracy': acc,'train_accuracy': float(self.acc_tr), # 'accuracy sd': acc_std,
                        'avg error': err, 'sd error': err_sd,
                        'avg error x': err_x, 'sd error x': err_x_sd,
                        'avg error y': err_y, 'sd error y': err_y_sd,
                        'avg error z': err_z, 'sd error z': err_z_sd}


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

        # create results csv
        if self.validation_runs > 1:
            for v in range(self.validation_runs):
                n_obs = len(self.preds[v])

                feat_vec = [self.feat_type] * n_obs
                rot_vec = [self.rotation_type] * n_obs
                valtype_vec = [self.val_type] * n_obs
                elecname_vec = [self.electrode_name] * n_obs
                px_vec = [self.n_points] * n_obs
                mea_ydim_vec = [self.mea_dim[0]] * n_obs
                mea_zdim_vec = [self.mea_dim[1]] * n_obs
                pitch_ydim_vec = [self.pitch[0]] * n_obs
                pitch_zdim_vec = [self.pitch[1]] * n_obs
                size_vec = [self.size] * n_obs
                val_run_vec = [v + 1] * n_obs

                cat = [self.cell_dict[cc] for cc in self.cat[v]]
                bin_cat = self.get_binary_cat(cat)
                preds = self.preds[v]
                loc = self.loc[v]
                rot = self.rot[v]
                errors = self.errors[v]
                errors_dim = self.errors_dim[v]

                d_obs = {'cell_type': cat, 'binary_cat': bin_cat,
                         'x': loc[:, 0], 'y': loc[:, 1], 'z': loc[:, 2],
                         'rot_x': rot[:,0], 'rot_y': rot[:,2], 'rot_z': rot[:,2],
                         'pred_x': preds[:, 0], 'pred_y': preds[:, 1], 'pred_z': preds[:, 2],
                         'err': errors, 'err_x': errors_dim[0], 'err_y': errors_dim[1], 'err_z': errors_dim[2],
                         'feat_type': feat_vec, 'rotation_type': rot_vec, 'val_type': valtype_vec,
                         'px': px_vec, 'y_pitch': pitch_ydim_vec, 'z_pitch': pitch_zdim_vec,
                         'MEA y dimension': mea_ydim_vec, 'MEA z dimension': mea_zdim_vec, 'val_run': val_run_vec,
                         'elec_name': elecname_vec, 'size': size_vec, 'id': np.arange(n_obs)}

                results_dir = join(self.model_path, 'results')
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                # df_obs.to_csv(join(results_dir, 'results.csv'))
                filen = open(join(results_dir, 'results.pkl'), 'wb')
                pickle.dump(d_obs, filen, protocol=2)
                filen.close()
        else:
            n_obs = len(self.preds)

            feat_vec = [self.feat_type] * n_obs
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
            bin_cat = self.get_binary_cat(cat)
            preds = self.preds
            loc = self.loc
            rot = self.rot
            errors = self.errors
            errors_dim = self.errors_dim

            d_obs = {'cell_type': cat, 'binary_cat': bin_cat,
                     'x': loc[:, 0], 'y': loc[:, 1], 'z': loc[:, 2],
                     'rot_x': rot[:,0], 'rot_y': rot[:,2], 'rot_z': rot[:,2],
                     'pred_x': preds[:, 0], 'pred_y': preds[:, 1], 'pred_z': preds[:, 2],
                     'err': errors, 'err_x': errors_dim[0], 'err_y': errors_dim[1], 'err_z': errors_dim[2],
                     'feat_type': feat_vec, 'rotation_type': rot_vec, 'val_type': valtype_vec,
                     'px': px_vec, 'y_pitch': pitch_ydim_vec, 'z_pitch': pitch_zdim_vec,
                     'MEA y dimension': mea_ydim_vec, 'MEA z dimension': mea_zdim_vec, 'val_run': val_run_vec,
                     'elec_name': elecname_vec, 'size': size_vec, 'id': np.arange(n_obs)}

            results_dir = join(self.model_path, 'results')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            # df_obs.to_csv(join(results_dir, 'results.csv'))
            filen = open(join(results_dir, 'results.pkl'), 'wb')
            pickle.dump(d_obs, filen, protocol=2)
            filen.close()



    def get_binary_cat(self, categories):
        binary_cat = []
        for i, cat in enumerate(categories):
            if cat in self.exc_categories:
                binary_cat.append('EXCIT')
            elif cat in self.inh_categories:
                binary_cat.append('INHIB')

        return np.array(binary_cat, dtype=str)

    def get_binary_cat_idx(self, categories):
        binary_cat = []
        for i, cat in enumerate(categories):
            if cat in self.exc_categories:
                binary_cat.append(int(np.argwhere(np.array(self.binary_cat) == 'EXCIT')))
            elif cat in self.inh_categories:
                binary_cat.append(int(np.argwhere(np.array(self.binary_cat) == 'INHIB')))

        return np.array(binary_cat)

    def get_mtype_cat_idx(self, categories, classes):
        m_cat = [int(np.argwhere(np.array(classes)==cat)) for cat in categories]
        return np.array(m_cat)

    def remondis(self):
        ''' clean up '''
        if self.tmp_data_dir.split('/')[-1].startswith('tmp'):
            shutil.rmtree(self.tmp_data_dir)
        else:
            raise UserWarning(self.tmp_data_dir + ' seems not to be a temporary directory. Did not remove it!')

if __name__ == '__main__':
    '''
        COMMAND-LINE 
        -f filename
        -feat feature type
        -val validation
        -cn cellnames
        -cs cellspecific
        -s  size
    '''

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        spike_folder = sys.argv[pos+1]  # file full path
    elif len(sys.argv) != 1:
        raise Exception('Provide spikes folder with argument -f')
    if '-feat' in sys.argv:
        pos = sys.argv.index('-feat')
        feat_type = sys.argv[pos+1]  # Na - NaRep - Rep - 3d
    else:
        feat_type = 'NaRep'
    if '-val' in sys.argv:
        pos = sys.argv.index('-val')
        val_type = sys.argv[pos+1]   # holdout - k-fold - hold-model-out - k-fold-model
    else:
        val_type = 'holdout'
    if '-modelout' in sys.argv:
        pos = sys.argv.index('-modelout')
        model_out = int(sys.argv[pos + 1])
    else:
        model_out = int(5)
    if '-n' in sys.argv:
        pos = sys.argv.index('-n')
        nsteps = int(sys.argv[pos+1])
    else:
        nsteps = 2000
    if '-cn' in sys.argv:
        pos = sys.argv.index('-cn')
        cell_names = sys.argv[pos+1]  # all - 'filename'
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
        size = sys.argv[pos+1]  # xs - s - m - l - xl
    else:
        size = 'l'
    if '-seed' in sys.argv:
        pos = sys.argv.index('-seed')
        seed = int(sys.argv[pos+1])  # xs - s - m - l - xl
    else:
        seed = int(2308)
    if len(sys.argv) == 1:
        print('Arguments: \n   -f full-path\n   -feat feature type: Na - Rep - NaRep - 3d\n   ' \
              '-val validation: holdout - k-fold - hold-model-out - k-fold-model\n   ' \
              '-cn cellnames: all - filename\n   -tcn train cellnames file\n   -vcn validation cellnames file' \
              '-s  size: xs - s - m - l - xl\n   -modelout model to hold out\n   -seed random seed')
    else:
        cv = SpikeConvNet(save=True, spike_folder=spike_folder, feat_type=feat_type, val_type=val_type, nsteps=nsteps,
                          cellnames=cell_names, size=size, model_out=model_out,
                          train_cell_names=train_cell_names, val_cell_names=val_cell_names, seed=seed)
