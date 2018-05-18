#!/usr/bin/env python
"""

Collection of functions for MEA stimulation

Conventions:
position = [um]
current = [nA]
voltage = [mV]

"""

import numpy as np
import copy

def get_elcoords(xoffset, dim, pitch, electrode_name, sortlist, **kwargs):
    """ Calculate electrode positions according to the arguments.
        MEA is placed in the y-z plane.

    Parameters:
    -----------
    xoffset : float
        Offset in x direction. If xoffset=0, all contact sites of the MEA 
        have x-coordinate = 0.
    dim : array_like
        Length-2 array, where dim[0] (dim[1]) specifies number of contacts in
        y-dimension (z-dimension)
    pitch : array_like
        Length-2 array, where pitch[0] (pitch[1]) specifies pitch of contacts in
        y-dimension (z-dimension)
    electrode_name : string
        Name of the electrode. Some electrodes have special arrangement 
        (checkerboard, hexagonal) and must be treaten in a special way.
    sortlist : array_like
        List for resorting electrodes. 

    Returns:
    --------
    el_pos_sorted : array_like
        sorted electrode site positions
    """
    if 'neuronexus-32' in electrode_name.lower():
        # calculate hexagonal order
        coldims = [10, 12, 10]
        if 'cut' in electrode_name.lower():
            coldims = [10, 10, 10]
        if sum(coldims) != np.prod(dim):
            raise ValueError('Dimensions in Neuronexus-32-channel probe do not match.')
        zshift = -pitch[1] * (coldims[1] - 1) / 2.
        x = np.array([0.] * sum(coldims))
        y = np.concatenate([[-pitch[0]] * coldims[0], [0.] * coldims[1], [pitch[0]] * coldims[2]])
        z = np.concatenate((np.arange(pitch[1] / 2., coldims[0] * pitch[1], pitch[1]),
                            np.arange(0., coldims[1] * pitch[1], pitch[1]),
                            np.arange(pitch[1] / 2., coldims[2] * pitch[1], pitch[1]))) + zshift
    elif 'neuropixels' in electrode_name.lower():
        if 'v1' in electrode_name.lower():
            # checkerboard structure
            x, y, z = np.mgrid[0:1,-(dim[0]-1)/2.:dim[0]/2.:1, -(dim[1]-1)/2.:dim[1]/2.:1]
            x=x+xoffset
            yoffset = np.array([pitch[0]/4.,-pitch[0]/4.]*(dim[1]/2)) 
            y=np.add(y*pitch[0],yoffset) #y*pitch[0]
            z=z*pitch[1]
        elif 'v2' in electrode_name.lower():
            # no checkerboard structure
            x, y, z = np.mgrid[0:1,-(dim[0]-1)/2.:dim[0]/2.:1, -(dim[1]-1)/2.:dim[1]/2.:1]
            x=x+xoffset
            y=y*pitch[0]
            z=z*pitch[1]
        else:
            raise NotImplementedError('This version of the NeuroPixels Probe is not implemented')
    else:
        x, y, z = np.mgrid[0:1, -(dim[0] - 1) / 2.:dim[0] / 2.:1, -(dim[1] - 1) / 2.:dim[1] / 2.:1]
        x = x + xoffset
        y = y * pitch[0]
        z = z * pitch[1]

    el_pos = np.concatenate((np.reshape(x, (x.size, 1)),
                             np.reshape(y, (y.size, 1)),
                             np.reshape(z, (z.size, 1))), axis=1)
    # resort electrodes in case
    el_pos_sorted = copy.deepcopy(el_pos)
    if sortlist is not None:
        for i, si in enumerate(sortlist):
            el_pos_sorted[si] = el_pos[i]

    return el_pos_sorted


