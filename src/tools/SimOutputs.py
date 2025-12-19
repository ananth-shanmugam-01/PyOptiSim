import numpy as np

class SimOutputs:
    def __init__(self):
        self.states       = dict()
        self.der_states  = dict()
        self.path_constraints = dict()
        self.auxiliary_outputs = dict()
        self.cost = []
        self.controls     = dict()
        self.parameters   = dict()
        self.mesh         = dict()
        
def createOutputDict(optiProblem, modelFun, Xs, Us, Gs) -> dict:
    
    simOut = SimOutputs()
    
    for ii in range(modelFun.states.num_x):
        simOut.states[modelFun.states.name[ii]] = np.array(optiProblem.value(Xs[ii,:]))
        
    for ii in range(modelFun.controls.num_u):
        simOut.controls[modelFun.controls.name[ii]] = np.array(optiProblem.value(Us[ii,:]))  
        
    for ii in range(modelFun.parameters.num_g):
        simOut.parameters[modelFun.parameters.name[ii]] = np.array(optiProblem.value(Gs[ii,:]))
        
    simOut.mesh = modelFun.mesh_points
    
    return simOut

def createDebugOutputDict(optiProblem, modelFun, Xs, Us, Gs) -> dict:
    
    simOut = SimOutputs()
    
    for ii in range(modelFun.states.num_x):
        simOut.states[modelFun.states.name[ii]] = np.array(optiProblem.debug.value(Xs[ii,:]))
        
    for ii in range(modelFun.controls.num_u):
        simOut.controls[modelFun.controls.name[ii]] = np.array(optiProblem.debug.value(Us[ii,:]))  
        
    for ii in range(modelFun.parameters.num_g):
        simOut.parameters[modelFun.parameters.name[ii]] = np.array(optiProblem.debug.value(Gs[ii,:]))

    rhs, L, path_constraints, auxiliary_outputs = modelFun.modelFunction(Xs, Us, Gs)
                                                
    for ii in range(modelFun.states.num_x):
        simOut.der_states[modelFun.states.der_names[ii]] = np.array(optiProblem.debug.value(rhs[ii,:]))

    if path_constraints.size(1) > 0:
        for ii in range(modelFun.path_constraints.num_path):
            simOut.path_constraints[modelFun.path_constraints.name[ii]] = np.array(optiProblem.debug.value(path_constraints[ii,:]))

    if auxiliary_outputs.size(1) > 0:
        for ii in range(modelFun.auxiliary_outputs.num_aux):
            simOut.auxiliary_outputs[modelFun.auxiliary_outputs.name[ii]] = np.array(optiProblem.debug.value(auxiliary_outputs[ii,:]))

    simOut.cost = np.array(optiProblem.debug.value(L))
    
    simOut.mesh = modelFun.mesh_points
    
    return simOut