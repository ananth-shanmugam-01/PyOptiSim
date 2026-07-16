import os
import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots as splt
from scipy.interpolate import PchipInterpolator

import tools.DecisionVariables as DecisionVariables
from model.BaseModel import BaseModel

# Transcription
import tools.OptiProblem as OptiProblem

# Post Processing
import tools.SimOutputs as SimOutputs

# Sim Debugging
from tools.DebugSim import DebugSim


""" 
Function to define the ordinary differential equations for the track model

To Run the test-harness:
python -m model.Track.TrackModel

"""

class TrackModel(BaseModel):

    def ProcessRawTrackData(self, trackFile: str, smoothing_factor: float = 1e5):
        """ Load Track Data from JSON and update model parameters, until then use the values directly"""

        trackData = pd.read_csv(trackFile)

        trackParams = dict()

        trackParams['Kt_smoothing_factor'] = smoothing_factor

        trackParams['xi'] = trackData['# x_m'].values
        trackParams['yi'] = trackData['y_m'].values

        dx = np.diff(trackParams['xi'], prepend=trackParams['xi'][0])
        dy = np.diff(trackParams['yi'], prepend=trackParams['yi'][0])
        dS = np.hypot(dx, dy)
        sLap = np.cumsum(dS)

        x_dot = np.gradient(trackParams['xi'], sLap, edge_order=2)
        y_dot = np.gradient(trackParams['yi'], sLap, edge_order=2)
        x_ddot = np.gradient(x_dot, sLap, edge_order=2)
        y_ddot = np.gradient(y_dot, sLap, edge_order=2)

        trackParams['Kt'] = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2)**1.5

        # Calculate mesh distance
        trackParams['sLap'] = np.cumsum(dS)

        # Calculate heading angle, correct for origin heading angle
        theta = np.cumsum(trackParams['Kt'] * dS)
        trackParams['aTheta'] = theta

        # Re-Calculate x and y positions based on corrected heading angle
        trackParams['xi'] = np.cumsum(dS * np.cos(trackParams['aTheta']))
        trackParams['yi'] = np.cumsum(dS * np.sin(trackParams['aTheta']))

        return self.settings.update({'track': trackParams})


    def createInitialSolution(self, trackParams=dict()):

        # Interpolate to Main Mesh
        initialSolution = dict()


        initialSolution["xi"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['xi'])(self.mesh_points)
        initialSolution["yi"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['yi'])(self.mesh_points)
        initialSolution["Kt"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['Kt'])(self.mesh_points)
        initialSolution["aTheta"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['aTheta'])(self.mesh_points)

        initialSolution["u"] = np.zeros(len(self.mesh_points))

        self.initialSolution = initialSolution


    def createModelFunction(self):

        # States
        Kt = ca.SX.sym('Kt') # Track Curvature (1/m) 
        self.states = DecisionVariables.addState(self.states, Kt, 'Kt', 'der_Kt', 1, (-0.5, 0.5), 3, (0, 0), self.initialSolution["Kt"])

        aTheta = ca.SX.sym('aTheta') # Track Heading Angle (rad)
        self.states = DecisionVariables.addState(self.states, aTheta, 'aTheta', 'der_aTheta', 1, (-np.inf, np.inf), 0, (0,0), self.initialSolution["aTheta"])

        xi = ca.SX.sym('xi') # Track X Position (m)
        self.states = DecisionVariables.addState(self.states, xi, 'xi', 'der_x', 100, (-np.inf, np.inf), 0, (0,0), self.initialSolution["xi"])

        yi = ca.SX.sym('yi') # Track Y Position (m)
        self.states = DecisionVariables.addState(self.states, yi, 'yi', 'der_y', 100, (-np.inf, np.inf), 0, (0,0), self.initialSolution["yi"])

        # Controls
        u = ca.SX.sym('u') # Curvature Smoothing Factor
        self.controls = DecisionVariables.addControl(self.controls, u, 'u', 1, (-1, 1), self.initialSolution["u"])

        # Parameters
        x_ref = ca.SX.sym('x_ref') # Reference X Position (m)
        self.parameters = DecisionVariables.addParameter(self.parameters, x_ref, 'x_ref', self.initialSolution["xi"])

        y_ref = ca.SX.sym('y_ref') # Reference Y Position (m)
        self.parameters = DecisionVariables.addParameter(self.parameters, y_ref, 'y_ref', self.initialSolution["yi"])

        c = ca.SX.sym('c') # Curvature Smoothing Factor - Scalar
        self.parameters = DecisionVariables.addParameter(self.parameters, c, 'c', self.settings['track']['Kt_smoothing_factor'])


        # Model Equations
        Sf = 1 # Scaling Factor = 1

        rhs = ca.SX.sym('rhs', self.states.num_x)
        rhs[0] = u
        rhs[1] = Kt
        rhs[2] = ca.cos(aTheta)
        rhs[3] = ca.sin(aTheta)

        cost = (x_ref - xi)**2 + (y_ref - yi)**2 + c * u**2

        # Model Function
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, cost, self.path_constraints.sym, self.auxiliary_outputs.sym],['x', 'u', 'g'], ['rhs', 'cost', 'path_constraints', 'auxiliary_outputs'])


if __name__ == "__main__":


    # Raw Track Data File Path
    fp = "/Users/ananthshanmugam/Desktop/GitHub/racetrack-database/racelines/Budapest.csv"
    # print(pd.read_csv(fp).columns)  # Display the column names of the CSV file


    # IPOPT Settings
    p_opts = {}
    s_opts = {"max_iter": 1000, 
            "tol" : 1e-6,
            "acceptable_tol": 1e-4,
            "constr_viol_tol": 1e-3,
            "compl_inf_tol": 1e-3,
            "nlp_scaling_method": 'gradient-based',}
    
    modelFun = TrackModel()

    modelFun.ProcessRawTrackData(fp, smoothing_factor=1e5)

    endPoint = modelFun.settings['track']['sLap'][-1]
    numIntervals = 400 # Number of Phases

    modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
    modelFun.createMesh(endPoint, numIntervals)
    modelFun.createInitialSolution(modelFun.settings['track'])
    modelFun.createModelFunction()

    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
    optiProblem.solver('ipopt', p_opts, s_opts)
    sol = optiProblem.solve()

    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, '/Users/ananthshanmugam/Desktop/SimResults/TrackMaking', 'Budapest.csv')

    # Load Results for Plotting
    results = pd.read_csv('/Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest.csv')

    # Plotting
    plt.style.use('science')
    plt.figure(figsize=(10, 6))
    plt.plot(results['sLap'], results['Kt'], label='Optimized Curvature', color='green')
    plt.plot(modelFun.settings['track']['sLap'], modelFun.settings['track']['Kt'], label='Original Curvature', color='orange', linestyle='--')
    plt.title('Track Curvature Optimization Results')
    plt.xlabel('Arc Length (m)')
    plt.ylabel('Curvature (1/m)')
    plt.legend()
    plt.grid()
    plt.show()