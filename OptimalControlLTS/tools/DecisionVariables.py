import numpy as np
import casadi as ca

class state:
    def __init__(self):
        self.sym       = []
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

class control:
    def __init__(self):
        self.sym    = []
        self.name   = []
        self.scale  = []
        self.lb     = []
        self.ub     = []
        self.num_u  = []
        self.u_init = []
        
class parameter:
    def __init__(self):
        self.sym    = []
        self.name   = []
        self.value  = []
        self.num_g  = []

def initialiseDecisionVariables():
    
    states = state()
    controls = control()
    parameters = parameter()

    return states, controls, parameters

def addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
    
    # Examples - states = tools.addState(states, n, 'n', 'der_n', 1, (1 1), 2, [0 0], [0], mesh.meshPoints)
    # BC - 0 - No BC, 1 - Initial Fixed, 2 - continuity, 3 - Initial and Terminal Fixed
    # Initial Solution to be interpolated outside
    
    states.sym        = ca.vertcat(states.sym, sym)
    states.name       = np.append(states.name, name)
    states.der_names  = np.append(states.der_names, der_name)
    states.scale      = np.append(states.scale, scale)
    states.lb         = np.append(states.lb, bounds[0])
    states.ub         = np.append(states.ub, bounds[1])
    states.BC         = np.append(states.BC, BC)
    states.BCini      = np.append(states.BCini, BC_Vals[0])
    states.BCend      = np.append(states.BCend, BC_Vals[1])
    states.x_init.append(np.array(initialSolution))
    states.num_x      = states.sym.size(1)
    
    return states

def addControl(controls, sym, name, scale, bounds, initialSolution):
    
    # Examples - states = tools.addState(states, n, 'n', 'der_n', 1, (1 1), 2, [0 0], [0], mesh.meshPoints)
    # BC - 0 - No BC, 1 - Initial Fixed, 2 - continuity, 3 - Initial and Terminal Fixed
    # Initial Solution to be interpolated outside
    
    controls.sym      = ca.vertcat(controls.sym, sym)
    controls.name     = np.append(controls.name, name)
    controls.scale    = np.append(controls.scale, scale)
    controls.lb       = np.append(controls.lb, bounds[0])
    controls.ub       = np.append(controls.ub, bounds[1])
    controls.u_init.append(np.array(initialSolution))
    controls.num_u    = controls.sym.size(1)
    
    return controls

def addParameter(parameters, sym, name, value):
    
    parameters.sym    = ca.vertcat(parameters.sym, sym)
    parameters.name   = np.append(parameters.name, name)
    parameters.value.append(np.array(value))
    parameters.num_g  = parameters.sym.size(1) 
    
    return parameters

#%% Testing

# states, controls, parameters = initialiseDecisionVariables()   

# velocity = ca.SX.sym('velocity')
# velocity_init = np.linspace(0, 10, 12) 
# states = addState(states, velocity, 'velocity', 'der_velocity', 1, (0, 100), 1, (10, 0), velocity_init)

# mass     = ca.SX.sym('mass')
# mass_init = np.ones(10) * 1 
# states = addState(states, mass, 'mass', 'der_mass', 1, (0, 20), 1, (10, 0), mass_init)
 