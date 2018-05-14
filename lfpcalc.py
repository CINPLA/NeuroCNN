'''

Simplified private functions from LFPy to compute ball-and-stick

'''
import numpy as np


def linesource_calc_case1(l_i, r2_i, h_i):
    """Calculates linesource contribution for case i"""
    bb = np.sqrt(h_i*h_i + r2_i) - h_i
    cc = np.sqrt(l_i*l_i + r2_i) - l_i
    dd = np.log(bb / cc)
    return dd


def linesource_calc_case2(l_ii, r2_ii, h_ii):
    """Calculates linesource contribution for case ii"""
    bb = np.sqrt(h_ii*h_ii + r2_ii) - h_ii
    cc = (l_ii + np.sqrt(l_ii*l_ii + r2_ii)) / r2_ii
    dd = np.log(bb * cc)
    return dd


def linesource_calc_case3(l_iii, r2_iii, h_iii):
    """Calculates linesource contribution for case iii"""
    bb = np.sqrt(l_iii*l_iii + r2_iii) + l_iii
    cc = np.sqrt(h_iii*h_iii + r2_iii) + h_iii
    dd = np.log(bb / cc)
    return dd


def deltaS_calc(xstart, xend, ystart, yend, zstart, zend):
    """Returns length of each segment"""
    deltaS = np.sqrt((xstart - xend)**2 + (ystart - yend)**2 +
                     (zstart-zend)**2)
    return deltaS


def h_calc(xstart, xend, ystart, yend, zstart, zend, deltaS, x, y, z):
    """Subroutine used by calc_lfp_*()"""
    aa = np.array([x - xend, y - yend, z-zend])
    bb = np.array([xend - xstart, yend - ystart, zend - zstart])
    cc = np.sum(aa*bb, axis=0)
    hh = cc / deltaS
    return hh


def r2_calc(xend, yend, zend, x, y, z, h):
    """Subroutine used by calc_lfp_*()"""
    r2 = (x-xend)**2 + (y-yend)**2 + (z-zend)**2 - h**2
    return abs(r2)


def r_soma_calc(xmid, ymid, zmid, x, y, z):
    """calculate the distance to soma midpoint"""
    r_soma = np.sqrt((x - xmid)**2 + (y - ymid)**2 + (z - zmid)**2)
    return r_soma