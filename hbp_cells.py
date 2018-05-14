#!/usr/bin/env python
from __future__ import division

'''
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads are unzipped in the folder 'cell_models'

The function compile_all_mechanisms most be ran once before any cell simulation
'''

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import LFPy
print('LFPy imported')
import neuron
print('neuron imported')
import MEAutility as MEA
import yaml
import json

from defaultconfig import *
if os.path.exists('./config.py'):
    from config import *


import time


neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    templatename = None
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue
    return templatename


def compile_all_mechanisms(model):
    """
    attempt to set up a folder with all unique mechanism mod files and compile them all.
    Assumes all HBP cell models are in a folder 'cell_models'
    """

    if not os.path.isdir(join(root_folder, 'cell_models', model, 'mods')):
        os.mkdir(join(root_folder, 'cell_models', model, 'mods'))

    neurons = [join(root_folder,'cell_models', model, f) for f in os.listdir(join(root_folder, 'cell_models', model))
               if f != 'mods']
    print neurons

    for nrn in neurons:
        for nmodl in glob(join(nrn, 'mechanisms', '*.mod')):
            print nmodl
            while not os.path.isfile(join(root_folder, 'cell_models', model, 'mods', os.path.split(nmodl)[-1])):
                print 'cp {} {}'.format(nmodl, join(root_folder, 'cell_models', model, 'mods'))
                os.system('cp {} {}'.format(nmodl, join(root_folder, 'cell_models', model, 'mods')))

    os.chdir('mods')
    os.system('nrnivmodl')


def return_cell(cell_folder, model_type, cell_name, end_T, dt, start_T,add_synapses=False):
    """
    Function to load Human Brain Project cell models
    :param cell_folder: where the cell model is stored
    :param cell_name: name of the cell
    :param end_T: simulation length [ms]
    :param dt: time resoultion [ms]
    :param start_T: simulation start time (recording starts at 0 ms)
    :return: LFPy cell object
    """
    cwd = os.getcwd()
    os.chdir(cell_folder)
    print "Simulating ", cell_name


    if model_type == 'bbp':
        neuron.load_mechanisms('../mods')

        f = file("template.hoc", 'r')
        templatename = get_templatename(f)
        f.close()

        f = file("biophysics.hoc", 'r')
        biophysics = get_templatename(f)
        f.close()

        f = file("morphology.hoc", 'r')
        morphology = get_templatename(f)
        f.close()

        #get synapses template name
        f = file(join("synapses", "synapses.hoc"), 'r')
        synapses = get_templatename(f)
        f.close()

        print('Loading constants')
        neuron.h.load_file('constants.hoc')
        print('...done.')
        if not hasattr(neuron.h, morphology):
            print 'loading morpho...'
            neuron.h.load_file(1, "morphology.hoc")
            print 'done.'

        if not hasattr(neuron.h, biophysics):
            neuron.h.load_file(1, "biophysics.hoc")

        if not hasattr(neuron.h, synapses):
            # load synapses
            neuron.h.load_file(1, join('synapses', 'synapses.hoc'))

        if not hasattr(neuron.h, templatename):
            print 'Loading template...'
            neuron.h.load_file(1, "template.hoc")
            print 'done.'

        morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

        # ipdb.set_trace()
        # Instantiate the cell(s) using LFPy
        print('Initialize cell...')
        cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                         templatefile=join('template.hoc'),
                         templatename=templatename,
                         templateargs=1 if add_synapses else 0,
                         tstop=end_T,
                         tstart=start_T,
                         dt=dt,
                         v_init=-70,
                         pt3d=True,
                         delete_sections=True,
                         verbose=True)
        print('...done.')

    os.chdir(cwd)
    return cell


def find_spike_idxs(v, thresh=-30):
    """
    :param v: membrane potential
    :return: Number of zero-crossings in the positive direction, i.e., number of spikes
    """
    spikes = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
    return spikes


def set_input(weight, dt, T, cell, delay, stim_length):
    """
    Set current input synapse in soma
    :param weight: strength of input current [nA]
    :param dt: time step of simulation [ms]
    :param T: Total simulation time [ms]
    :param cell: cell object from LFPy
    :param delay: when to start the input [ms]
    :param stim_length: duration of injected current [ms]
    :return: NEURON vector of input current, cell object, and synapse
    """

    tot_ntsteps = int(round(T / dt + 1))

    I = np.ones(tot_ntsteps) * weight
    #I[stim_idxs] = weight
    noiseVec = neuron.h.Vector(I)
    syn = None
    for sec in cell.allseclist:
        if 'soma' in sec.name():
            # syn = neuron.h.ISyn(0.5, sec=sec) 
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = stim_length
    syn.delay = delay  # cell.tstartms
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn


def run_cell_model(cell_model, model_type, sim_folder, figure_folder, cell_model_id):
    """
    Run simulation and adjust input strength to have between number of spikes between num_spikes and 3*num_spikes
    :param cell_model: Name of cell model (should correspond to name of subfolder in folder "cell_models"
    :param figure_folder: Folder to save figures in
    :param cell_model_id: number for random seed
    """

    cell_name = os.path.split(cell_model)[-1]
    print sim_folder

    if not os.path.isfile(join(sim_folder, ('i_spikes_%s.npy' % cell_name))) and \
            not os.path.isfile(join(sim_folder, ('v_spikes_%s.npy' % cell_name))):

        np.random.seed(123 * cell_model_id)
        T = 1200
        dt = 2 ** -5
        cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

        delay = 200
        stim_length = 1000
        weight = 0.23
        # weight = -1.25

        num_spikes = 0
        spikes = []

        cut_out = [2. / dt, 5. / dt]
        num_to_save = 10

        i = 0

        while not num_to_save < num_spikes <= num_to_save * 3:
            noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)

            cell.simulate(rec_imem=True)

            t = cell.tvec
            v = cell.somav
            t = t
            v = v

            # ipdb.set_trace()

            spikes = find_spike_idxs(v[int(cut_out[0]):-int(cut_out[1])])
            spikes = list(np.array(spikes) + cut_out[0])
            num_spikes = len(spikes)

            print "Input weight: ", weight, " - Num Spikes: ", num_spikes
            if num_spikes >= num_to_save * 3:
                weight *= 0.75
            elif num_spikes <= num_to_save:
                weight *= 1.25

            i += 1

            if i >= 10:
                sys.exit()

        t = t[0:(int(cut_out[0]) + int(cut_out[1]))] - t[int(cut_out[0])]
        i_spikes = np.zeros((num_to_save, cell.totnsegs, len(t)))
        v_spikes = np.zeros((num_to_save, len(t)))

        for idx, spike_idx in enumerate(spikes[1:num_to_save+1]):
            spike_idx = int(spike_idx)
            v_spike = v[spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spike = cell.imem[:, spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spikes[idx, :, :] = i_spike
            v_spikes[idx, :] = v_spike

        if not os.path.isdir(sim_folder):
            os.makedirs(sim_folder)
        np.save(join(sim_folder, 'i_spikes_%s.npy' % cell_name), i_spikes)
        np.save(join(sim_folder, 'v_spikes_%s.npy' % cell_name), v_spikes)

        return cell

    else:
        print 'Cell has already be simulated. Using stored membrane currents'
        np.random.seed(123 * cell_model_id)
        T = 1200
        dt = 2 ** -5
        cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

        return cell

def activate_synapses(cell):
    '''activate inserted synapses
    '''

    # get pre_mtypes and their id
    pre_mtypes = cell.template.synapses.pre_mtypes
    # loop over mtypes
    for i in range(int(cell.template.synapses.pre_mtypes.size())):
        pre_mtype_id = int(cell.template.synapses.pre_mtypes.x[i])
        # pre_mtype_freqs = $o1.synapses.pre_mtype_freqs
        # pre_mtype_name = $o1.synapses.id_mtype_map.o(pre_mtype_id).s
        active_pre_mtypes = cell.template.synapses.active_pre_mtypes
        # activate specific pre_mtype synapse by setting the value of active_pre_mtypes.x[pre_mtype_id] to 1
        active_pre_mtypes.x[pre_mtype_id]=1
        # update_synapses() in NEURON
        # cell.cell.synapses.update_synapses(synapse_plot_dummy)
        cell.template.synapses.update_synapses()



def run_cell_model_poisssyn(cell_model, model_type, sim_folder, figure_folder, cell_model_id):
    """
    Run simulation and adjust input strength to have between number of spikes between num_spikes and 3*num_spikes
    :param cell_model: Name of cell model (should correspond to name of subfolder in folder "cell_models"
    :param figure_folder: Folder to save figures in
    :param cell_model_id: number for random seed
    """
    cell_name = os.path.split(cell_model)[-1]

    if not os.path.isfile(join(sim_folder, ('i_spikes_%s.npy' % cell_name))) and \
            not os.path.isfile(join(sim_folder, ('v_spikes_%s.npy' % cell_name))):

        np.random.seed(123 * cell_model_id)
        T = 1500
        dt = 2 ** -5
        cell = return_cell(cell_model, model_type, cell_name, T, dt, 0,add_synapses=True)
        
        activate_synapses(cell)

        # delay = 200
        # stim_length = 1000
        # weight = 0.23
        # # weight = -1.25

        num_spikes = 0
        spikes = []

        cut_out = [2. / dt, 5. / dt]
        num_to_save = 10

        while not num_to_save < num_spikes <= num_to_save * 3:
            # noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)
            cell.simulate(rec_imem=True)

            t = cell.tvec
            v = cell.somav
            t = t  # [-cut_off_idx:] #- t[-cut_off_idx]
            v = v  # [-cut_off_idx:]
            spikes = find_spike_idxs(v)
            num_spikes = len(spikes)
            print('Number of spikes %d' % num_spikes)

        t = t[0:(int(cut_out[0]) + int(cut_out[1]))] - t[int(cut_out[0])]
        i_spikes = np.zeros((num_to_save, cell.totnsegs, len(t)))
        v_spikes = np.zeros((num_to_save, len(t)))
        for idx, spike_idx in enumerate(spikes[1:num_to_save+1]):
            v_spike = v[spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spike = cell.imem[:, spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spikes[idx, :, :] = i_spike
            v_spikes[idx, :] = v_spike

        if not os.path.isdir(sim_folder):
            os.makedirs(sim_folder)
        np.save(join(sim_folder, 'i_spikes_%s.npy' % cell_name), i_spikes)
        np.save(join(sim_folder, 'v_spikes_%s.npy' % cell_name), v_spikes)
    else:
        print 'Cell has already be simulated. Using stored membrane currents'


def calc_extracellular(cell_model, model_type, save_sim_folder,
                       load_sim_folder, rotation, cell_model_id, elname, nobs, position=None):
    """
    Loads data from previous cell simulation, and use results to generate arbitrary number of spikes above a certain
    noise level.
    :param cell_model: Folder where cell model is
    :param save_sim_folder: Folder to save data
    :param load_sim_folder: Folder to load neuron sim currents
    :param rotation: type of rotation to apply to neuron morphologies
    :param cell_model_id: number to make random seed
    :return:
    """
    sim_folder = join(save_sim_folder, rotation)

    np.random.seed(123 * cell_model_id)
    dt = 2**-5
    T = 1

    cell_name = os.path.split(cell_model)[-1]
    cell_save_name = cell_name

    cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

    # Load data from previous cell simulation
    i_spikes = np.load(join(load_sim_folder, 'i_spikes_%s.npy' % cell_name))
    v_spikes = np.load(join(load_sim_folder, 'v_spikes_%s.npy' % cell_name))

    cell.tvec = np.arange(i_spikes.shape[-1]) * dt

    save_spikes = []
    save_pos = []
    save_rot = []
    save_offs = []
    target_num_spikes = int(nobs)
    noise_level = 30  # uV peak-to-peak
    i = 0

    # specify MEA
    MEAname = elname

    # load MEA info
    with open(join(root_folder,'electrodes', MEAname+'.json')) as meafile:
        elinfo = json.load(meafile)
    
    # specify number of points for average EAP on each site
    n = 1  # 10 # 50
    elinfo.update({'n_points': n})

    # Create save folder
    # Create directory with target_spikes and date
    save_folder = join(sim_folder, 'e_%d_%dpx_%.1fum_%.1fum_%s_%s' % (target_num_spikes, n,
                                                                      elinfo['pitch'][0],elinfo['pitch'][1],
                                                                      MEAname, time.strftime("%d-%m-%Y")))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Check if already existing
    if os.path.isfile(join(save_folder, 'e_spikes_%d_%s_%s.npy' %
            (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y")))) and \
        os.path.isfile(join(save_folder, 'e_pos_%d_%s_%s.npy' %
            (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y")))) and \
        os.path.isfile(join(save_folder, 'e_rot_%d_%s_%s.npy' %
            (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y")))):
        print 'Cell ', cell_save_name, ' extracellular spikes have already been simulated and saved'
    else:
        print 'Cell ', cell_save_name, ' extracellular spikes to be simulated'

        x_plane = 0.
        pos = MEA.get_elcoords(x_plane,**elinfo)

        elec_x = pos[:, 0]
        elec_y = pos[:, 1]
        elec_z = pos[:, 2]

        N = np.empty((pos.shape[0], 3))
        for i in xrange(N.shape[0]):
            N[i, ] = [1, 0, 0]  # normal vec. of contacts

        # Add square electrodes (instead of circles)
        if n > 1:
            electrode_parameters = {
                'sigma': 0.3,  # extracellular conductivity
                'x': elec_x,  # x,y,z-coordinates of contact points
                'y': elec_y,
                'z': elec_z,
                'n': n,
                'r': elinfo['r'],
                'N': N,
                'contact_shape': elinfo['shape']
            }
        else:
            electrode_parameters = {
                'sigma': 0.3,  # extracellular conductivity
                'x': elec_x,  # x,y,z-coordinates of contact points
                'y': elec_y,
                'z': elec_z
            }
            
        overhang = 30. # um in y, and z direction
        y_lim = [min(elec_y)-elinfo['pitch'][0]/2.-overhang,max(elec_y)+elinfo['pitch'][0]/2.+overhang]
        z_lim = [min(elec_z)-elinfo['pitch'][1]/2.-overhang,max(elec_z)+elinfo['pitch'][1]/2.+overhang]
        x_lim = [10., 80.]

        ignored=0
        saved = 0

        while len(save_spikes) < target_num_spikes:
            if i > 1000 * target_num_spikes:
                print "Gave up finding spikes above noise level for %s" % cell_name
                break
            spike_idx = np.random.randint(0, i_spikes.shape[0])  # Each cell has several spikes to choose from
            cell.imem = i_spikes[spike_idx, :, :]
            cell.somav = v_spikes[spike_idx, :]

            espikes, pos, rot, offs = return_extracellular_spike(cell, cell_name, model_type, electrode_parameters,
                                                                 [x_lim,y_lim,z_lim], rotation, pos=position)
            if (np.ptp(espikes, axis=1) >= noise_level).any():
                # print "Big spike!"
                save_spikes.append(espikes)
                save_pos.append(pos)
                save_rot.append(rot)
                save_offs.append(offs)
                plot_spike = False
                print 'Cell: ' + cell_name + ' Progress: [' + str(len(save_spikes)) + '/' + str(target_num_spikes) + ']'
                saved += 1
            else:
                pass


            i += 1

        save_spikes = np.array(save_spikes)
        save_pos = np.array(save_pos)
        save_rot = np.array(save_rot)
        save_offs = np.array(save_offs)

        np.save(join(save_folder, 'e_spikes_%d_%s_%s.npy' % (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y"))),
                save_spikes)
        np.save(join(save_folder, 'e_pos_%d_%s_%s.npy' % (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y"))),
                save_pos)
        np.save(join(save_folder, 'e_rot_%d_%s_%s.npy' % (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y"))),
                save_rot)
        if not os.path.isfile(join(save_folder, 'e_elpts_%d.npy' % target_num_spikes)):
            np.save(join(save_folder, 'e_elpts_%d.npy' % target_num_spikes),
                    save_offs)

        # Log information: (consider xml)
        with open(join(save_folder, 'e_info_%d_%s_%s.yaml' % (target_num_spikes, cell_save_name, time.strftime("%d-%m-%Y"))),
                'w') as f:
            # create dictionary for yaml file
            data_yaml = {'General': {'cell name': cell_name, 'target spikes': target_num_spikes, 
                                     'noise level': noise_level, 'NEURON': neuron.h.nrnversion(1), 
                                     'LFPy': LFPy.__version__ , 'dt': dt},
                        'Electrodes': elinfo,
                        'Location': {'z_lim': z_lim,'y_lim': y_lim, 'x_lim': x_lim, 'rotation': rotation}
                        }
            yaml.dump(data_yaml, f, default_flow_style=False)

        print save_spikes.shape
        
def get_physrot_specs(cell_name, model):
    '''  Return physrot specifications for cell_type
    '''
    if model == 'bbp':
        polarlim = {'BP': [0.,15.],
                    'BTC': None, # [0.,15.],
                    'ChC': None, # [0.,15.],
                    'DBC': None, # [0.,15.],
                    'LBC': None, # [0.,15.],
                    'MC': [0.,15.],
                    'NBC': None,
                    'NGC': None,
                    'SBC': None,
                    'STPC': [0.,15.],
                    'TTPC1': [0.,15.],
                    'TTPC2': [0.,15.],
                    'UTPC': [0.,15.]}
        # how it's implemented, the NMC y axis points into the pref_orient direction after rotation
        pref_orient = {'BP': [0.,0.,1.],
                       'BTC': None, # [0.,0.,1.],
                       'ChC': None, # [0.,0.,1.],
                       'DBC': None, # [0.,0.,1.],
                       'LBC': None, # [0.,0.,1.],
                       'MC': [0.,0.,1.],
                       'NBC': None,
                       'NGC': None,
                       'SBC': None,
                       'STPC': [0.,0.,1.],
                       'TTPC1': [0.,0.,1.],
                       'TTPC2': [0.,0.,1.],
                       'UTPC': [0.,0.,1.]}
        return polarlim[cell_name.split('_')[1]], pref_orient[cell_name.split('_')[1]]
    elif model=='hay':
        return [0, 15], [0., 0., 1.]
    elif model=='almog':
        return [0, 15], [0., 0., 1.]

def get_exprot_specs(cell_name, model):
    '''  Return physrot specifications for cell_type
    '''
    if model == 'bbp':
        polarlim = {'BP': [0.,15.],
                    'BTC': None, # [0.,15.],
                    'ChC': None, # [0.,15.],
                    'DBC': None, # [0.,15.],
                    'LBC': None, # [0.,15.],
                    'MC': [0.,15.],
                    'NBC': None,
                    'NGC': None,
                    'SBC': None,
                    'STPC': [0.,15.],
                    'TTPC1': [0.,15.],
                    'TTPC2': [0.,15.],
                    'UTPC': [0.,15.]}
        # how it's implemented, the NMC y axis points into the pref_orient direction after rotation
        pref_orient = {'BP': [ 0.66653247, 0.,  0.745476 ], # -41.8 degree rotated
                       'BTC': None, # [0.,0.,1.],
                       'ChC': None, # [0.,0.,1.],
                       'DBC': None, # [0.,0.,1.],
                       'LBC': None, # [0.,0.,1.],
                       'MC': [ 0.66653247, 0.,  0.745476 ],
                       'NBC': None,
                       'NGC': None,
                       'SBC': None,
                       'STPC': [ 0.66653247, 0.,  0.745476 ],
                       'TTPC1': [ 0.66653247, 0.,  0.745476 ],
                       'TTPC2': [ 0.66653247, 0.,  0.745476 ],
                       'UTPC': [ 0.66653247, 0.,  0.745476 ]}
        return polarlim[cell_name.split('_')[1]], pref_orient[cell_name.split('_')[1]]
    elif model=='hay':
        return [0, 15], [ 0.66653247, 0.,  0.745476 ]
    elif model=='almog':
        return [0, 15], [ 0.66653247, 0.,  0.745476 ]



def return_extracellular_spike(cell, cell_name, model_type, electrode_parameters, limits, rotation, pos=None):
    """
    Calculate extracellular spike at tetrode at random position relative to cell
    :param cell: cell object from LFPy
    :param cell_name: name of cell model (string)
    :param electrode_parameters: parameters to initialize LFPy.RecExtElectrode
    :param limits: boundaries for neuron locations
    :param rotation: 'Norot', 'Xrot', '3drot', 'physrot' - random rotation to apply to the neuron
    :param figure_folder: folder to save the figure to
    :param plot_spike: boolean value. Should we plot the spike?
    :return: Extracellular spike at tetrode contacts
    """

    def get_xyz_angles(R):
        '''
        R = R_z.R_y.R_x
        '''
        rot_x = np.arctan2(R[2,1],R[2,2])
        rot_y = np.arcsin(-R[2,0])
        rot_z = np.arctan2(R[1,0],R[0,0])
        # rotation = {
        #     'x' : rot_x,
        #     'y' : rot_y,
        #     'z' : rot_z,
        # }
        return rot_x,rot_y,rot_z

    def get_rnd_rot_Arvo():
        gamma = np.random.uniform(0,2.*np.pi)
        rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])
        x = np.random.uniform(size=2)
        v = np.array([np.cos(2.*np.pi*x[0])*np.sqrt(x[1]),
                      np.sin(2.*np.pi*x[0])*np.sqrt(x[1]),
                      np.sqrt(1-x[1])])
        H = np.identity(3)-2.*np.outer(v,v)
        M = -np.dot(H,rotation_z)
        return M

    def check_solidangle(matrix,pre,post,polarlim):
        ''' check whether matrix rotates pre into polarlim region around post
        '''
        postest = np.dot(matrix,pre)
        c=np.dot(post/np.linalg.norm(post),postest/np.linalg.norm(postest))
        if np.cos(np.deg2rad(polarlim[1])) <= c <= np.cos(np.deg2rad(polarlim[0])):
            return True
        else:
            return False

    electrodes = LFPy.RecExtElectrode(cell, **electrode_parameters)

    '''Rotate neuron'''
    if rotation == 'Norot':
        # orientate cells in z direction
        if model_type == 'bbp':
            x_rot = np.pi / 2.
            y_rot = 0
            z_rot = 0
    elif rotation == '3drot':
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2. # align neuron with z axis
            y_rot_offset = 0 # align neuron with z axis
            z_rot_offset = 0 # align neuron with z axis

        x_rot, y_rot, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot + z_rot_offset

    elif rotation == 'physrot':
        polarlim, pref_orient  = get_physrot_specs(cell_name, model_type)
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2. # align neuron with z axis
            y_rot_offset = 0 # align neuron with z axis
            z_rot_offset = 0 # align neuron with z axis
        while True:
            R = np.array(get_rnd_rot_Arvo())
            if polarlim is None or pref_orient is None:
                valid = True
            else:
                valid = check_solidangle(R,[0.,0.,1.],pref_orient,polarlim)
            if valid:
                x_rot,y_rot,z_rot = get_xyz_angles(R)
                x_rot = x_rot + x_rot_offset
                y_rot = y_rot + y_rot_offset
                z_rot = z_rot + z_rot_offset
                break
    else:
        x_rot = 0
        y_rot = 0
        z_rot = 0

    '''Move neuron randomly'''
    x_rand = np.random.uniform(limits[0][0], limits[0][1])
    y_rand = np.random.uniform(limits[1][0], limits[1][1])
    z_rand = np.random.uniform(limits[2][0], limits[2][1])


    if pos == None:
        cell.set_pos(x_rand, y_rand, z_rand)
    else:
        cell.set_pos(pos[0], pos[1], pos[2])
    cell.set_rotation(x=x_rot, y=y_rot, z=z_rot)
    pos = [x_rand, y_rand, z_rand]
    rot = [x_rot, y_rot, z_rot]

    electrodes.calc_lfp()

    # Reverse rotation to bring cell back into initial rotation state
    rev_rot = [-rot[e] for e in range(len(rot))]
    cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order='zyx')

    return 1000 * electrodes.LFP, pos, rot, electrodes.offsets


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':

    if len(sys.argv) == 1:
        raise RuntimeError("Wrong usage. Give argument 'compile' to compile mechanisms," +
                           " and cell name, model_type, cell id, only_intra (bool), rotation, probe to simulate cell")
    elif sys.argv[1] == 'compile':
            compile_all_mechanisms(sys.argv[2])
            sys.exit(0)
    elif len(sys.argv) == 8:
        cell_folder, model, numb, only_intracellular, rotation, probe, nobs = sys.argv[1:]
        only_intracellular = str2bool(only_intracellular)


    extra_sim_folder = join(data_dir, 'spikes', model)
    vm_im_sim_folder = join(data_dir, 'spikes', model, 'Vm_Im')

    print vm_im_sim_folder

    print cell_folder, model, numb, only_intracellular, rotation
    cell = run_cell_model(cell_folder, model, vm_im_sim_folder, vm_im_sim_folder, int(numb))

    if not only_intracellular:
        print 'ROTATION type: ', rotation
        calc_extracellular(cell_folder, model, extra_sim_folder, vm_im_sim_folder, rotation, int(numb), probe, nobs)


