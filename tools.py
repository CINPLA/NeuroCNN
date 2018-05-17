from __future__ import print_function
# Helper functions

import numpy as np
import os
from os.path import join

def load(filename):
    '''Generic loading of cPickled objects from file'''
    import pickle
    
    filen = open(filename,'rb')
    obj = pickle.load(filen)
    filen.close()
    return obj

def yaml_load(filename):
    '''Generic saving of dictionary to yaml file'''
    import yaml
    
    filen = open(filename, 'r')
    dic=yaml.load(filen)
    filen.close()
    return dic


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

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
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def load_EAP_data(spike_folder, cell_names, all_categories,samples_per_cat=None):

    print("Loading spike data ...")
    spikelist = [f for f in os.listdir(spike_folder) if f.startswith('e_spikes') and  any(x in f for x in cell_names)]
    loclist = [f for f in os.listdir(spike_folder) if f.startswith('e_pos') and  any(x in f for x in cell_names)]
    rotlist = [f for f in os.listdir(spike_folder) if f.startswith('e_rot') and  any(x in f for x in cell_names)]

    cells, occurrences = np.unique(sum([[f.split('_')[4]]*int(f.split('_')[2]) for f in spikelist],[]), return_counts=True)
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
    return np.array(spikes_list), np.array(loc_list), np.array(rot_list), np.array(category_list, dtype=str), \
        np.array(etype_list, dtype=str), np.array(morphid_list, dtype=int), loaded_categories


def load_validation_data(validation_folder,load_mcat=False):
    print("Loading validation spike data ...")

    spikes = np.load(join(validation_folder, 'val_spikes.npy'))  # [:spikes_per_cell, :, :]
    feat = np.load(join(validation_folder, 'val_feat.npy'))  # [:spikes_per_cell, :, :]
    locs = np.load(join(validation_folder, 'val_loc.npy'))  # [:spikes_per_cell, :]
    rots = np.load(join(validation_folder, 'val_rot.npy'))  # [:spikes_per_cell, :]
    cats = np.load(join(validation_folder, 'val_cat.npy'))
    if load_mcat:
        mcats = np.load(join(validation_folder, 'val_mcat.npy'))
        print("Done loading spike data ...")
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats), np.array(mcats)
    else:
        print("Done loading spike data ...")
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats)


def load_vm_im(vm_im_folder, cell_names, all_categories):
    
    print("Loading membrane potential and currents data ...")
    vmlist = [f for f in os.listdir(vm_im_folder) if f.startswith('v_spikes') and  any(x in f for x in cell_names)]
    imlist = [f for f in os.listdir(vm_im_folder) if f.startswith('i_spikes') and  any(x in f for x in cell_names)]
    cat_list = [f.split('_')[3] for f in vmlist]
    entries_in_category = {cat: cat_list.count(cat) for cat in all_categories if cat in all_categories}
    # print "Number of cells in each category", entries_in_category

    vmlist = sorted(vmlist)
    imlist = sorted(imlist)

    vm_list = []
    im_list = []
    category_list = []

    loaded_categories = []
    ignored_categories = []

    for idx, f in enumerate(vmlist):
        category = f.split('_')[3]

        if category in all_categories:
            vm = np.load(join(vm_im_folder, f)) # [:spikes_per_cell, :, :]
            vm_list.append(vm)
            im = np.load(join(vm_im_folder, imlist[idx])) # [:spikes_per_cell, :]
            im_list.append(im)
            loaded_categories.append(category)
        else:
            ignored_categories.append(category)

    print("Done loading spike data ...")
    print("Loaded categories", loaded_categories)
    print("Ignored categories", ignored_categories)
    return np.array(vm_list), im_list, np.array(loaded_categories, dtype=str)


def get_EAP_features(EAP,feat_list,dt=None,EAP_times=None,threshold_detect=5.,normalize=False):
    ''' extract features specified in feat_list from EAP
    '''
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
        min_idx = np.array([np.unravel_index(EAP[i, e].argmin(), EAP[i, e].shape)[0] for e in
                                       range(EAP.shape[1])])
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
            
        amps[i, :] = np.array([EAP[i, e, max_idx[e]]-EAP[i, e, min_idx[e]] for e in range(EAP.shape[1])])
        # If below 'detectable threshold, set amp and width to 0
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

            # features['fwhm'][i,:] = np.array([(len(t)+1)*dt+(fwhm_V[e]-EAP[i,e,t[0]-1])/(EAP[i,e,t[0]]-EAP[i,e,t[0]-1])*dt -(fwhm_V[e]-EAP[i,e,t[-1]])/(EAP[i,e,min(t[-1]+1,EAP.shape[2]-1)]-EAP[i,e,t[-1]])*dt if len(t)>0 else np.infty for e,t in enumerate(id_trough)])

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
    binary_cat = []
    for i, cat in enumerate(categories):
        if cat in excit:
            binary_cat.append('EXCIT')
        elif cat in inhib:
            binary_cat.append('INHIB')

    return np.array(binary_cat, dtype=str)


def get_cat_from_label_idx(label_cat, categories):
    cat = []
    if len(label_cat) > 1:
        for i, cc in enumerate(label_cat):
            cat.append(categories[cc])
    elif len(label_cat) == 1:
        cat.append(categories[label_cat])

    return cat

def get_cat_from_hot_label(hot_label_cat, categories):
    cat = []
    if len(hot_label_cat.shape) == 2:
        for i, cc in enumerate(hot_label_cat):
            cat_id = int(np.where(cc == 1)[0])
            cat.append(categories[cat_id])
    elif len(hot_label_cat.shape) == 1:
        cat_id = int(np.where(hot_label_cat == 1)[0])
        cat.append(categories[cat_id])

    return cat


def convert_metype(mapfile,typ):
    ''' convert an m/e-type specifying string to integer or vice versa
    according to assignment in mapfile, or dictionary
    '''
    if type(filename)==str:
        mapdict = dict(np.loadtxt(mapfile,dtype=str))
    elif type(mapfile)==dict:
        mapdict = mapfile
    else:
        raise TypeError('Cannot handle type of mapfile.')

    if type(typ)==str and typ in mapdict.values():
        return int(float(mapdict.keys()[mapdict.values().index(typ)]))
    elif type(typ)==int:
        return mapdict.get(str(typ))
    else:
        raise ValueError('Can\'t handle m/e-type of the cell.')
