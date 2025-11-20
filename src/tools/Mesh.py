# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:43:25 2024

@author: admin
"""

import numpy as np

class mesh:
    def __init__( self, endPoint, meshPhases, collocation_degree, tau ):
        self.endPoint       = endPoint 
        self.meshPhases     = meshPhases
        self.meshSize       = endPoint/meshPhases
        self.collocation_degree = collocation_degree
        self.tau            = tau

        meshInterval = np.linspace(0, self.endPoint, self.meshPhases+1)  
        
        remesh = np.zeros((self.meshPhases, self.collocation_degree+1))
        
        for i in range(len(meshInterval)-1):
            dMesh = self.meshSize
            remesh[i,0] = meshInterval[i]
            for ii in range(self.collocation_degree):
                remesh[i, ii+1] = remesh[i,0] + dMesh*self.tau[ii]
        
        self.mesh = np.vstack( (np.reshape(remesh, (np.size(remesh),1)), meshInterval[-1]) )



