import casadi as ca
import numpy as np

import src.tools.Mesh as Mesh
import src.tools.DecisionVariables as DecisionVariables

class model:
    def __init__(self):
        
        # Mesh Properties
        self.mesh_size = []
        self.mesh_numIntervals = []
        self.mesh_points   = [] # equivalent to sLap
        
        # Decision Variables
        self.states, self.controls, self.parameters = DecisionVariables.initialiseDecisionVariables()
        
        # Model Outputs
        self.cost         = []
        self.path_constraints = []
        self.modelFunction    = []
                       
    def createLagrangeCoefficients(self, collocation_degree, collocation_strategy):
        
        self.collocation_degree = collocation_degree
        self.collocation_strategy = collocation_strategy
        self.collocation_tau = np.array(ca.collocation_points(collocation_degree, collocation_strategy))
        self.collocation_C, self.collocation_D, self.collocation_B = ca.collocation_coeff(self.collocation_tau)
        
    def createMesh(self, endPoint, numIntervals):
        
        meshObject = Mesh.mesh( endPoint, numIntervals, self.collocation_degree, self.collocation_tau )
        self.mesh_size = meshObject.meshSize
        self.mesh_numIntervals = numIntervals
        self.mesh_points = meshObject.mesh
    
    def createModelFunction(self):

        # States
        # addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        velocity = ca.SX.sym('velocity')
        velocity_init = 10 * np.ones(len(self.mesh_points)) 
        self.states = DecisionVariables.addState(self.states, velocity, 'velocity', 'der_velocity', 10, (0, 100), 4, (10, 0), velocity_init)
        
        mass     = ca.SX.sym('mass')
        mass_init = 1 * np.ones(len(self.mesh_points)) 
        self.states = DecisionVariables.addState(self.states, mass, 'mass', 'der_mass', 1, (0, 10), 1, (1, 0), mass_init)
        
        # Controls
        thrust = ca.SX.sym('thrust')
        thrust_init = np.ones(len(self.mesh_points))
        self.controls = DecisionVariables.addControl(self.controls, thrust, 'thrust',10, (0, 20), thrust_init)
        
        # Parameters
        g = ca.SX.sym('g')
        g_mesh = 9.81 * np.ones(len(self.mesh_points))
        self.parameters = DecisionVariables.addParameter(self.parameters, g, 'g', g_mesh)
        
        # Assemble Model 
        c = 0.05 # Consumption factor
        Sf = 1/velocity
        
        # Model Dynamics
        rhs = ca.SX.sym('rhs', self.states.num_x)
        rhs[0] = Sf * (g - thrust/mass)
        rhs[1] = Sf * (-c * thrust)
        
        # Model Penalties
        L = Sf
        
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, L],['x', 'u', 'g'], ['rhs', 'L'])

    def factory():
        
        endPoint = 50 # [m] Mesh Distance
        numIntervals = 15 # Number of Phases
        
        modelFun = model()
        modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
        modelFun.createMesh(endPoint, numIntervals)
        modelFun.createModelFunction()
        
        return modelFun

    def unitTestModel():   
        
        endPoint = 50 # [m] Mesh Distance
        numIntervals = 15 # Number of Phases
        
        modelFun = model()
        modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
        modelFun.createMesh(endPoint, numIntervals)
        modelFun.createModelFunction()
        
        # Test Function
        [x_dot, cost] = modelFun.modelFunction([8, 1], 4, 1);
        print(x_dot)
        print(cost)
        
        return modelFun
