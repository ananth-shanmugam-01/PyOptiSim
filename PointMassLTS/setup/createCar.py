# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:31:54 2024

@author: admin
"""
import json 

def createCar(setupFile):

    setupFile = open(setupFile)
    
    data = json.load( setupFile ) # Hooray, we have parameters
    
    return data