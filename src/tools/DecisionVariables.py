import numpy as np
import casadi as ca

class state:
    def __init__(self):
        self.sym       = []
        self._symbols  = {}
        self.derivatives = {}
        self.name       = []
        self.der_names  = []
        self.scale      = []
        self.lb         = []
        self.ub         = []
        self.BC         = []
        self.BCini      = []
        self.BCend      = []
        self.x_init     = []
        self.num_x      = []

    def __getitem__(self, name):
        return self._symbols[name]

    def __contains__(self, name):
        return name in self._symbols

class control:
    def __init__(self):
        self.sym    = []
        self._symbols = {}
        self.name   = []
        self.scale  = []
        self.lb     = []
        self.ub     = []
        self.num_u  = []
        self.u_init = []

    def __getitem__(self, name):
        return self._symbols[name]

    def __contains__(self, name):
        return name in self._symbols
        
class parameter:
    def __init__(self):
        self.sym    = []
        self._symbols = {}
        self.name   = []
        self.value  = []
        self.num_g  = []

    def __getitem__(self, name):
        return self._symbols[name]

    def __contains__(self, name):
        return name in self._symbols

class pathConstraint:
    def __init__(self):
        self.sym    = []
        self.name   = []
        self.scale  = []
        self.lb     = []
        self.ub     = []
        self.num_path  = []

class auxiliaryOutput:
    def __init__(self):
        self.sym    = []
        self.name   = []
        self.num_aux  = []

class penalty:
    def __init__(self):
        self.sym       = []
        self._symbols  = {}
        self.name      = np.array([])
        self.num_penalty = 0

    def __getitem__(self, name):
        return self._symbols[name]

    def __contains__(self, name):
        return name in self._symbols

def initialiseDecisionVariables():
    
    states = state()
    controls = control()
    parameters = parameter()
    pathConstraints = pathConstraint()
    auxiliaryOutputs = auxiliaryOutput()

    return states, controls, parameters, pathConstraints, auxiliaryOutputs

def addState(states, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
    
    # BC - 0 - No BC, 1 - Initial Fixed, 2 - continuity, 3 - Initial and Terminal Fixed
    # Initial Solution to be interpolated outside
    
    sym = ca.SX.sym(name)

    if name in states:
        raise ValueError(f"State '{name}' has already been added")

    states._symbols[name] = sym
    states.sym        = ca.vertcat(states.sym, sym)
    states.name       = np.append(states.name, name)
    states.der_names  = np.append(states.der_names, der_name)
    states.scale      = np.append(states.scale, scale )
    states.lb         = np.append(states.lb, bounds[0])
    states.ub         = np.append(states.ub, bounds[1])
    states.BC         = np.append(states.BC, BC)
    states.BCini      = np.append(states.BCini, BC_Vals[0])
    states.BCend      = np.append(states.BCend, BC_Vals[1])
    states.x_init.append(np.array(initialSolution))
    states.num_x      = states.sym.size(1)
    
    return states

def addControl(controls, name, scale, bounds, initialSolution):
    
    # BC - 0 - No BC, 1 - Initial Fixed, 2 - continuity, 3 - Initial and Terminal Fixed
    # Initial Solution to be interpolated outside
    
    sym = ca.SX.sym(name)

    if name in controls:
        raise ValueError(f"Control '{name}' has already been added")

    controls._symbols[name] = sym
    controls.sym      = ca.vertcat(controls.sym, sym)
    controls.name     = np.append(controls.name, name)
    controls.scale    = np.append(controls.scale, scale )
    controls.lb       = np.append(controls.lb, bounds[0])
    controls.ub       = np.append(controls.ub, bounds[1])
    controls.u_init.append(np.array(initialSolution))
    controls.num_u    = controls.sym.size(1)
    
    return controls

def addParameter(parameters, name, value):
    
    sym = ca.SX.sym(name)

    if name in parameters:
        raise ValueError(f"Parameter '{name}' has already been added")

    parameters._symbols[name] = sym
    parameters.sym    = ca.vertcat(parameters.sym, sym)
    parameters.name   = np.append(parameters.name, name)
    parameters.value.append(np.array(value))
    parameters.num_g  = parameters.sym.size(1) 
    
    return parameters

def addPathConstraint(pathConstraints, sym, name, scale, bounds):
    
    pathConstraints.sym      = ca.vertcat(pathConstraints.sym, sym)
    pathConstraints.name     = np.append(pathConstraints.name, name)
    pathConstraints.scale    = np.append(pathConstraints.scale, scale )
    pathConstraints.lb       = np.append(pathConstraints.lb, bounds[0])
    pathConstraints.ub       = np.append(pathConstraints.ub, bounds[1])
    pathConstraints.num_path  = pathConstraints.sym.size(1)
    
    return pathConstraints

def addAuxiliaryOutput(auxiliaryOutputs, sym, name):
    
    auxiliaryOutputs.sym      = ca.vertcat(auxiliaryOutputs.sym, sym)
    auxiliaryOutputs.name     = np.append(auxiliaryOutputs.name, name)
    auxiliaryOutputs.num_aux  = auxiliaryOutputs.sym.size(1)
    
    return auxiliaryOutputs

def addPenalty(penalties, sym, name):
    """Register a named scalar stage-cost or penalty expression."""

    if name in penalties:
        raise ValueError(f"Penalty '{name}' has already been added")

    penalties._symbols[name] = sym
    penalties.sym = ca.vertcat(penalties.sym, sym)
    penalties.name = np.append(penalties.name, name)
    penalties.num_penalty = penalties.sym.size(1)

    return penalties
