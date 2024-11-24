# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:46:29 2024

@author: admin
"""

import numpy as np
import scipy.interpolate as interp

def createTrack(trackFile, nMesh):

    track = np.loadtxt(trackFile)

    sLap = np.linspace(track[0,0], track[-1,0], nMesh)

    kt = interp.interp1d(track[:,0], track[:,1])

    kt   = np.maximum(kt(sLap), 0.00001)
    
    return sLap, kt