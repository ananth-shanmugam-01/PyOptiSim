import numpy as np

class SimOutputs:
    def __init__(self):
        self.states       = dict()
        self.controls     = dict()
        self.parameters   = dict()
        self.mesh         = dict()
        
def createOutputDict(optiProblem, modelFun, Xs, Us, Gs):
    
    simOut = SimOutputs()
    
    for ii in range(modelFun.states.num_x):
        simOut.states[modelFun.states.name[ii]] = np.array(optiProblem.value(Xs[ii,:]))
        
    for ii in range(modelFun.controls.num_u):
        simOut.controls[modelFun.controls.name[ii]] = np.array(optiProblem.value(Us[ii,:]))  
        
    for ii in range(modelFun.parameters.num_g):
        simOut.parameters[modelFun.parameters.name[ii]] = np.array(optiProblem.value(Gs[ii,:]))
        
    simOut.mesh = modelFun.mesh_points
    
    return simOut
