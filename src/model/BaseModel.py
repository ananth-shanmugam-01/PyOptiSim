"""
Base Class for Model contains common properties and methods
- initialising decision variables
- Create Lagrange Coefficients
- Create Mesh
- Bring them all together into a factory
- create initial solution (empty here, to be overridden in child classes)
- create model function (empty here, to be overridden in child classes)

"""
# Base Packages
import casadi as ca
import numpy as np
# Custom Packages

import tools.Mesh as Mesh
import tools.DecisionVariables as DecisionVariables

class BaseModel:
    def __init__(self):
        
        # Mesh Properties
        self.mesh_size = []
        self.mesh_numIntervals = []
        self.mesh_points   = [] # equivalent to sLap

        # Settings
        self.settings = dict()
        
        # Decision Variables
        self.states, self.controls, self.parameters, self.path_constraints, self.auxiliary_outputs = DecisionVariables.initialiseDecisionVariables()
        
        # Initial Solution
        self.initialSolution = dict()

        # Model Outputs
        self.cost         = []
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

    def factory(self):
        """
        Factory Method to return an instance of the model
        """
        modelFun = self.model
        modelFun.createLagrangeCoefficients( 3, 'legendre' ) # collocation degree and strategy, fixed for now
        modelFun.createMesh( self.meshEndPoint, self.meshNumIntervals )
        modelFun.createInitialSolution( self.meshEndPoint, self.meshNumIntervals )
        modelFun.createModelFunction()

        return modelFun
    
    def createModelFunction(self):
        """
        To be overridden in child classes
        """
        pass

    def factory():
        """
        To be overridden in child classes
        """
        pass