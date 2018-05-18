from __future__ import print_function
# Helper functions

import numpy as np
import os
from os.path import join

def load(filename):
    """ Generic loading of cPickled objects from file
    
    Parameters:
    -----------
    filename, str 
        Name of file to open

    Returns:
    -------
    obj, object
        object returned by pickle.load()
    """
    import pickle
    
    filen = open(filename,'rb')
    obj = pickle.load(filen)
    filen.close()
    return obj

def yaml_load(filename):
    """ Generic loading of dictionary from yaml file

    Parameters:
    -----------
    filename, str
        Name of file to load

    Returns:
    -------
    dic, dict
        loaded dictionary
    """
    import yaml
    
    filen = open(filename, 'r')
    dic=yaml.load(filen)
    filen.close()
    return dic


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """ Savitzky-Golay filtering
    (function from SciPy Cookbook)

    Parameters:
    -----------
    y : array_like
        Signal data to be filtered
    window_size : int
        Length of the filter window
    order : int
        Order of polynomial
    deriv : int (optional, default 0)
        Order of derivative to compute
    rate : float (optional,default 1)
        Reciprocal spacing of window data points.
    Returns:
    --------
    filtered signal, array_like
    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] \
                for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def load_EAP_data(spike_folder, cell_names, all_categories,\
                  samples_per_cat=None):
    """ Loading extracellular action potential data from files
    
    Parameters:
    -----------
    spike_folder : str
        Path to folder containing the data.
    cell_names : list
        List of cell names for which data is loaded.
    all_categories : list
        Cell categories to be taken into account.
    samples_per_cat : int (optional, default None)
        Number of samples per category to be loaded. If ``None``,
        all available samples are loaded

    Returns:
    --------
    spikes_list : array_like
        Loaded EAPs
    loc_list : array_like
        Positions of the neurons evoking the loaded EAPs.
    rot_list : array_like
        Rotations (angles) of the neurons evoking the loaded EAPs.
    category_list : array_like (dtype=str)
        Categories of the neurons evoking the loaded EAPs.
    etype_list : array_like (dtype=str)
        Electrical type of the neurons evoking the loaded EAPs.
    morphid_list : array_like
        Morphological id of the neurons evoking the loaded EAPs.
    loaded_categories : list
        List of loaded categories.
    """
    print("Loading spike data ...")
    spikelist = [f for f in os.listdir(spike_folder) if f.startswith('e_spikes')\
                 and  any(x in f for x in cell_names)]
    loclist = [f for f in os.listdir(spike_folder) if f.startswith('e_pos')\
               and  any(x in f for x in cell_names)]
    rotlist = [f for f in os.listdir(spike_folder) if f.startswith('e_rot')\
               and  any(x in f for x in cell_names)]

    cells, occurrences = np.unique(sum([[f.split('_')[4]]*int(f.split('_')[2])\
                                for f in spikelist],[]), return_counts=True)
    occ_dict = dict(zip(cells,occurrences))
    spikes_list = []

    loc_list = []
    rot_list = []
    category_list = []
    etype_list = []
    morphid_list = []

    spikelist = sorted(spikelist)
    loclist = sorted(loclist)
    rotlist = sorted(rotlist)

    loaded_categories = set()
    ignored_categories = set()

    for idx, f in enumerate(spikelist):
        category = f.split('_')[4]
        samples = int(f.split('_')[2])
        if samples_per_cat is not None:
            samples_to_read = int(float(samples)/occ_dict[category]*samples_per_cat)
        else:
            samples_to_read = samples
        etype = f.split('_')[5]
        morphid = f.split('_')[6]
        print('loading ', samples_to_read , ' samples for cell type: ', f)
        if category in all_categories:
            spikes = np.load(join(spike_folder, f)) # [:spikes_per_cell, :, :]
            spikes_list.extend(spikes[:samples_to_read])
            locs = np.load(join(spike_folder, loclist[idx])) # [:spikes_per_cell, :]
            loc_list.extend(locs[:samples_to_read])
            rots = np.load(join(spike_folder, rotlist[idx])) # [:spikes_per_cell, :]
            rot_list.extend(rots[:samples_to_read])
            category_list.extend([category] * samples_to_read)
            etype_list.extend([etype] * samples_to_read)
            morphid_list.extend([morphid] * samples_to_read)
            loaded_categories.add(category)
        else:
            ignored_categories.add(category)

    print("Done loading spike data ...")
    print("Loaded categories", loaded_categories)
    print("Ignored categories", ignored_categories)
    return np.array(spikes_list), np.array(loc_list), np.array(rot_list),\
        np.array(category_list, dtype=str), np.array(etype_list, dtype=str),\
        np.array(morphid_list, dtype=int), loaded_categories


def load_validation_data(validation_folder,load_mcat=False):
    """ Load validation data from files

    Parameters:
    -----------
    validation_folder : str
        Path to folder containing the validation data.
    load_mcat : bool (optional, default False)
        Whether to load the information about the morphological type (m-type)
        of the neurons.

    Returns:
    --------
    spikes : array_like
        Loaded EAPs.
    feat : array_like
        Extracted EAP features.
    locs : array_like
        Positions of neurons evoking the EAP.
    rots : array_like
        Rotations (angles) of neurons evoking the EAP.
    cats : array_like
        Categories of neurons evoking the EAP.
    mcats : array_like (optional)
        Morphological categories are only returne if load_mcat=True.
    """
    print("Loading validation spike data ...")

    spikes = np.load(join(validation_folder, 'val_spikes.npy'))  # [:spikes_per_cell, :, :]
    feat = np.load(join(validation_folder, 'val_feat.npy'))  # [:spikes_per_cell, :, :]
    locs = np.load(join(validation_folder, 'val_loc.npy'))  # [:spikes_per_cell, :]
    rots = np.load(join(validation_folder, 'val_rot.npy'))  # [:spikes_per_cell, :]
    cats = np.load(join(validation_folder, 'val_cat.npy'))
    if load_mcat:
        mcats = np.load(join(validation_folder, 'val_mcat.npy'))
        print("Done loading spike data ...")
        return np.array(spikes), np.array(feat), np.array(locs),\
            np.array(rots), np.array(cats), np.array(mcats)
    else:
        print("Done loading spike data ...")
        return np.array(spikes), np.array(feat), np.array(locs),\
            np.array(rots), np.array(cats)


def get_EAP_features(EAP,feat_list,dt=None,EAP_times=None,threshold_detect=5.,normalize=False):
    """ Extract features specified in feat_list from EAP

    Parameters:
    -----------
    EAP : array_like
        Array of EAPs. Can be of shape (N_timepoints,),
        (N_electrodes,N_timepoints), or (N_spikes,N_electrodes,N_timepoints)
    feat_list : list
        List of features to extract. Possible values are:
        'A' - peak-to-peak amplitude
        'W' - peak-to-peak width
        'F' - Full-width half-maximum of negative peak
        'R' - ratio between maximum and minimum peak on each MEA site
        'S' - speed, the delay of each sites minimum peak compared to the 
              overall minimum peak on the MEA.
        'Aids' - indices of negative and positive peak
        'Fids' - indices of FWHM
        'FV'   - FWHM voltage
        'Na'   - negative peak
        'Rep'  - positive peak after negative one
    dt : float (optional, default None)
        Step size of EAP times. If ``None`` and EAP_times is used.  
        Either of the two has to be specified.
    EAP_times : array_like (optional, default None)
        Array of EAP times (len(EAP_times)=N_timepoints). If ``None``, 
        dt parameter is used. Either of the two has to be specified.
    threshold_detect : float (optional, default=5.)
        Peak-to-peak amplitude threshold below which features are neglected.
    normalized : bool (optional, default False)
        If True, EAPs of each spike are normalized to largest negative peak on 
        the overall MEA. 
    Returns:
    --------
    features : dict
        Dictionary of extracted features.
    """
    reference_mode = 't0'
    if EAP_times is not None and dt is not None:
        test_dt = (EAP_times[-1]-EAP_times[0])/(len(EAP_times)-1)
        if dt != test_dt:
            raise ValueError('EAP_times and dt do not match.')
    elif EAP_times is not None:
        dt = (EAP_times[-1]-EAP_times[0])/(len(EAP_times)-1)
    elif dt is not None:
        EAP_times = np.arange(EAP.shape[-1])*dt
    else:
        raise NotImplementedError('Please, specify either dt or EAP_times.')

    if len(EAP.shape)==1:
        EAP = np.reshape(EAP,[1,1,-1])
    elif len(EAP.shape)==2:
        EAP = np.reshape(EAP,[1,EAP.shape[0],EAP.shape[1]])
    if len(EAP.shape)!= 3:
        raise ValueError('Cannot handle EAPs with shape',EAP.shape)

    if normalize:
        signs = np.sign(np.min(EAP.reshape([EAP.shape[0],-1]),axis=1))
        norm = np.abs(np.min(EAP.reshape([EAP.shape[0],-1]),axis=1))
        EAP = np.array([EAP[i]/n if signs[i]>0 else EAP[i]/n-2. for i,n in enumerate(norm)])

    features = {}
    
    amps = np.zeros((EAP.shape[0], EAP.shape[1]))
    na_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    rep_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'W' in feat_list:
        features['widths'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'F' in feat_list:
        features['fwhm'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'R' in feat_list:
        features['ratio'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'S' in feat_list:
        features['speed'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Aids' in feat_list:
        features['Aids'] = np.zeros((EAP.shape[0], EAP.shape[1],2),dtype=int)
    if 'Fids' in feat_list:
        features['Fids'] = []
    if 'FV' in feat_list:
        features['fwhm_V'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Na' in feat_list:
        features['na'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Rep' in feat_list:
        features['rep'] = np.zeros((EAP.shape[0], EAP.shape[1]))


    for i in range(EAP.shape[0]):
        # For AMP feature
        min_idx = np.array([np.unravel_index(EAP[i, e].argmin(),\
                            EAP[i, e].shape)[0] for e in range(EAP.shape[1])])
        max_idx = np.array([np.unravel_index(EAP[i, e, min_idx[e]:].argmax(),
                                                        EAP[i, e, min_idx[e]:].shape)[0]
                                       + min_idx[e] for e in range(EAP.shape[1])])
        # for na and rep
        min_elid, min_idx_na = np.unravel_index(EAP[i].argmin(), EAP[i].shape)
        max_idx_rep = EAP[i,min_elid, min_idx_na:].argmax() + min_idx_na
        na_peak[i, :] = EAP[i, :, min_idx_na]
        rep_peak[i, :] = EAP[i, :, max_idx_rep]


        if 'Aids' in feat_list:
            features['Aids'][i]=np.vstack((min_idx, max_idx)).T
            
        amps[i, :] = np.array([EAP[i, e, max_idx[e]]-EAP[i, e, min_idx[e]]\
                               for e in range(EAP.shape[1])])
        # If below 'detectable threshold, set amp to zero and width to EAP length
        if normalize:
            too_low = np.where(amps[i, :] < threshold_detect/norm[i])
        else:
            too_low = np.where(amps[i, :] < threshold_detect)
        amps[i, too_low] = 0

        if 'R' in feat_list:
            min_id_ratio = np.array([np.unravel_index(EAP[i, e, min_idx_na:].argmin(),
                                                      EAP[i, e, min_idx_na:].shape)[0]
                                     + min_idx_na for e in range(EAP.shape[1])])
            max_id_ratio = np.array([np.unravel_index(EAP[i, e, min_idx_na:].argmax(),
                                                 EAP[i, e, min_idx_na:].shape)[0]
                                + min_idx_na for e in range(EAP.shape[1])])
            features['ratio'][i, :] = np.array([np.abs(EAP[i, e, max_id_ratio[e]])/
                                                np.abs(EAP[i, e, min_id_ratio[e]])
                                                for e in range(EAP.shape[1])])
            # If below 'detectable threshold, set amp and width to 0
            too_low = np.where(amps[i, :] < threshold_detect)
            features['ratio'][i, too_low] = 1
        if 'S' in feat_list:
            features['speed'][i, :] = np.array((min_idx - min_idx_na)*dt)
            features['speed'][i, too_low] = min_idx_na*dt
           
        if 'W' in feat_list:
            features['widths'][i, :] = np.abs(EAP_times[max_idx] - EAP_times[min_idx])
            features['widths'][i, too_low] = EAP.shape[2] * dt # EAP_times[-1]-EAP_times[0]

        if 'F' in feat_list:
            min_peak = np.min(EAP[i],axis=1)
            if reference_mode == 't0':
                # reference voltage is zeroth voltage entry
                fwhm_ref = np.array([EAP[i,e,0] for e in range(EAP.shape[1])])
            elif reference_mode == 'maxd2EAP':
                # reference voltage is taken at peak onset
                # peak onset is defined as id of maximum 2nd derivative of EAP
                peak_onset = np.array([np.argmax(savitzky_golay(EAP[i,e],5,2,deriv=2)[:min_idx[e]])
                                       for e in range(EAP.shape[1])])
                fwhm_ref = np.array([EAP[i,e,peak_onset[e]] for e in range(EAP.shape[1])])
            else:
                raise NotImplementedError('Reference mode ' + reference_mode + ' for FWHM calculation not implemented.')
            fwhm_V = (fwhm_ref + min_peak)/2. 
            id_trough = [np.where(EAP[i,e]<fwhm_V[e])[0] for e in range(EAP.shape[1])]
            if 'Fids' in feat_list:
                features['Fids'].append(id_trough)
            if 'FV' in feat_list:
                features['fwhm_V'][i,:]= fwhm_V

            # linear interpolation due to little number of data points during peak

            # features['fwhm'][i,:] = np.array([(len(t)+1)*dt+(fwhm_V[e]-EAP[i,e,t[0]-1])\
            #             /(EAP[i,e,t[0]]-EAP[i,e,t[0]-1])*dt -(fwhm_V[e]-EAP[i,e,t[-1]])\
            #             /(EAP[i,e,min(t[-1]+1,EAP.shape[2]-1)]-EAP[i,e,t[-1]])*dt\
            #             if len(t)>0 else np.infty for e,t in enumerate(id_trough)])

            # no linear interpolation
            features['fwhm'][i,:] = [(id_trough[e][-1] - id_trough[e][0])*dt if len(id_trough[e])>1 \
                                     else EAP.shape[2] * dt for e in range(EAP.shape[1])]
            features['fwhm'][i, too_low] = EAP.shape[2] * dt # EAP_times[-1]-EAP_times[0]

    if 'A' in feat_list:
        features.update({'amps': amps})
    if 'Na' in feat_list:
        features.update({'na': na_peak})
    if 'Rep' in feat_list:
        features.update({'rep': rep_peak})

    return features


def get_binary_cat(categories, excit, inhib):
    """ Get binary category from m-type category
    
    Parameters:
    -----------
    categories, array_like
        M-type categories
    excit : array_like
        List of m-types being excitatory cells.
    inhib : array_like
        List of m-types being inhibitory cells.

    Returns:
    --------
    binary_cat : array_like
        Binary categories of m-type categories in categories parameter.
    """
    binary_cat = []
    for i, cat in enumerate(categories):
        if cat in excit:
            binary_cat.append('EXCIT')
        elif cat in inhib:
            binary_cat.append('INHIB')

    return np.array(binary_cat, dtype=str)
