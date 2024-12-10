import casadi as ca
import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy import integrate
import math

import matplotlib.pyplot as plt

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
        
        # Initial Solution
        self.initialSolution = dict()

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
    
    def loadTrack(trackFile):
        df = pd.read_csv(trackFile)       
        x = df["# x_m"]
        y = df["y_m"]
        
        ds = np.sqrt( np.power(np.gradient(x),2) + np.power(np.gradient(y),2) )
        sLap = np.cumsum(ds)
        
        return df, sLap               
    
    def createInitialSolution(self, trackDataFrame):
        
        x = trackDataFrame["# x_m"]
        y = trackDataFrame["y_m"]
        
        ds = np.sqrt( np.power(np.gradient(x),2) + np.power(np.gradient(y),2) )
        sLap = np.cumsum(ds)
        
        # Hyper Sample Track Mesh
        sLap_fine = np.linspace(0,sLap[-1],1000)
        x = np.interp(sLap_fine, sLap, x)
        y = np.interp(sLap_fine, sLap, y)
        
        # Create Spline Objects
        x_spline = UnivariateSpline(sLap_fine, x, s=0.01)
        x_der_spline = x_spline.derivative(n=1)
        x_der_der_spline = x_spline.derivative(n=2)
        
        y_spline = UnivariateSpline(sLap_fine, y, s=0.01)
        y_der_spline = y_spline.derivative(n=1)
        y_der_der_spline = y_spline.derivative(n=2)
        
        x_dot       = x_der_spline(sLap_fine)
        x_ddot      = x_der_der_spline(sLap_fine)
        y_dot       = y_der_spline(sLap_fine)
        y_ddot      = y_der_der_spline(sLap_fine)
        
        # Calculate Track Curvature
        Curv = (x_dot* y_ddot - x_ddot*y_dot)/((x_dot**2 + y_dot**2 )**1.5)
        
        trackData = dict() # save in same interpolation scheme 
        
        trackData["sLap"] = (self.mesh_points)
        
        trackData["x"]    = UnivariateSpline(sLap_fine, x, s=0.0005)(self.mesh_points)
        trackData["y"]    = UnivariateSpline(sLap_fine, y, s=0.0005)(self.mesh_points)
        trackData["curv"] = UnivariateSpline(sLap_fine, Curv, s=0.0005)(self.mesh_points)
        trackData["theta"] = integrate.cumtrapz(trackData["curv"], self.mesh_points, initial=0)
        
        trackData["heading_origin"] = math.atan( y_dot[0] / x_dot[0] )
        trackData["corrected_theta"] = trackData["theta"] + trackData["heading_origin"]
        
        # Interpolate to Main Mesh 
        initialSolution = dict()
        initialSolution["x"]        = trackData["x"]
        initialSolution["y"]        = trackData["y"]
        initialSolution["curv"]     = trackData["curv"]
        initialSolution["theta"]    = trackData["theta"]

        self.initialSolution = initialSolution
        
        # # Plots to Verify Quality of Pre-Processing
        
        # plt.plot(sLap_fine, Curv,'-.',label='Raw')
        # plt.plot(sLap_fine, trackData["Curv"],label='Smooth')
        # plt.xlabel("sLap [m]")
        # plt.ylabel("Kt [1/m]")
        # plt.legend()
        # plt.title('Track Curvature Generation')
        # plt.show()
        
        # plt.plot( trackData["sLap"], trackData["theta"] * 180/3.14)
        # plt.plot( trackData["sLap"], trackData["corrected_theta"]* 180/3.14)
        # plt.xlabel("sLap [m]")
        # plt.ylabel("theta [rad]")
        # plt.title("Track Heading Angle")
        # plt.grid()
        # plt.legend(['theta', 'corrected_theta'])
        # plt.show()        


    def createModelFunction(self):

        # States
        # addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        curv = ca.SX.sym('curv')
        self.states = DecisionVariables.addState(self.states, curv, 'curv', 'der_curv', 1, (-3, 3), 3, (10, 0), self.initialSolution["curv"] )

        theta = ca.SX.sym('theta')
        self.states = DecisionVariables.addState(self.states, theta, 'theta', 'der_theta', 1, (-3*math.pi, 3*math.pi), 3, (10, 0), self.initialSolution["theta"] )

        xi = ca.SX.sym('xi')
        self.states = DecisionVariables.addState(self.states, xi, 'xi', 'der_xi', 1, (-math.inf, math.inf), 0, (10, 0), self.initialSolution["x"] )
        
        yi = ca.SX.sym('yi')
        self.states = DecisionVariables.addState(self.states, yi, 'yi', 'der_yi', 1, (-math.inf, math.inf), 0, (10, 0), self.initialSolution["y"] )    
        
        # Controls
        u = ca.SX.sym('u')
        u_init = 0 * np.ones(len(self.mesh_points))
        self.controls = DecisionVariables.addControl(self.controls, u, 'u',1, (-math.inf, math.inf), u_init)
        
        # Parameters       
        xc = ca.SX.sym('xc')
        self.parameters = DecisionVariables.addParameter(self.parameters, xc, 'xc', self.initialSolution["x"])
        
        yc = ca.SX.sym('yc')
        self.parameters = DecisionVariables.addParameter(self.parameters, yc, 'yc', self.initialSolution["y"])
        
        # Assemble Model 
        c = 2 # Smooth Factor
        
        # Model Dynamics
        rhs = ca.SX.sym('rhs', self.states.num_x)
        rhs[0] = u
        rhs[1] = curv
        rhs[2] = math.cos(theta)
        rhs[3] = math.sin(theta)
        
        # Model Penalties
        L = (xc - xi)**2 + (yc - yi)**2 + (c*u)**2
        
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, L],['x', 'u', 'g'], ['rhs', 'L'])

    def factory(trackDataFrame, sLap):

        modelFun = model()
        
        numIntervals = 500 # Number of Phases
        endPoint = sLap[-1]        
        
        modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
        modelFun.createMesh(endPoint, numIntervals)
        modelFun.createInitialSolution(trackDataFrame)
        modelFun.createModelFunction()
        
        return modelFun
    
    def createResultPlots(SimOut):

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot( SimOut.mesh , SimOut.states["velocity"], '-o')
        ax1.set_title('Velocity Trajectory')

        ax2.plot( SimOut.mesh , SimOut.states["mass"], '-o')
        ax2.set_title('Mass State Trajectory')

        ax3.plot( SimOut.mesh , SimOut.controls["thrust"], '-o')
        ax3.set_title('Thrust Control Trajectory')

        plt.show()

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
