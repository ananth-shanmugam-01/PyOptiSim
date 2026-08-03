import os
import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots as splt
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import UnivariateSpline
from scipy import integrate
from scipy.signal import savgol_filter

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

    def ProcessRawTrackData(self, trackFile: str, smoothing_factor: float = 1e2):
        """ Load Track Data from JSON and update model parameters, until then use the values directly"""

        trackData = pd.read_csv(trackFile)

        trackParams = dict()

        trackParams['Kt_smoothing_factor'] = smoothing_factor

        # Raw centerline
        x_raw = trackData["# x_m"].to_numpy()
        y_raw = trackData["y_m"].to_numpy()

        x_raw = x_raw - x_raw[0]  # Shift x to start from 0
        y_raw = y_raw - y_raw[0]  # Shift y to start from 0
        
        # Calculate Arc-length parameter from raw points
        dx = np.gradient(x_raw)
        dy = np.gradient(y_raw)
        ds = np.hypot(dx, dy)
        arc_length = np.cumsum(ds)

        # Resample arc-length to a uniform grid
        arc_length = np.linspace(arc_length[0], arc_length[-1], 2000)
        x_raw = PchipInterpolator(np.cumsum(ds), x_raw)(arc_length)
        y_raw = PchipInterpolator(np.cumsum(ds), y_raw)(arc_length)

        # Create UnivariateSpline objects for x and y
        sx = UnivariateSpline(arc_length, x_raw, s=1)
        sy = UnivariateSpline(arc_length, y_raw, s=1)

        dx_ds = sx.derivative(1)(arc_length)
        dx_dds = sx.derivative(2)(arc_length)
        dy_ds = sy.derivative(1)(arc_length)
        dy_dds = sy.derivative(2)(arc_length)

        # Curvature from spline derivatives
        Kt = (dx_ds * dy_dds - dy_ds * dx_dds) / (dx_ds**2 + dy_ds**2) ** 1.5

        # Heading angle from derivative direction
        aTheta = integrate.cumulative_trapezoid(Kt, arc_length, initial=0)
        aTheta = aTheta + np.arctan2(dy_ds[0], dx_ds[0])  # Adjust initial heading based on the first point

        x = integrate.cumulative_trapezoid(np.cos(aTheta), arc_length, initial=0)
        y = integrate.cumulative_trapezoid(np.sin(aTheta), arc_length, initial=0)

        trackParams["sLap"] = arc_length
        trackParams["xi"] = x
        trackParams["yi"] = y
        trackParams["Kt"] = Kt
        trackParams["aTheta"] = aTheta

        return self.settings.update({'track': trackParams})


    def createInitialSolution(self, trackParams=dict()):

        # Interpolate to Main Mesh
        initialSolution = dict()


        initialSolution["xi"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['xi'])(self.mesh_points)
        initialSolution["yi"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['yi'])(self.mesh_points)
        initialSolution["Kt"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['Kt'])(self.mesh_points)
        initialSolution["aTheta"] = PchipInterpolator(self.settings['track']['sLap'], trackParams['aTheta'])(self.mesh_points)

        initialSolution["u"] = np.zeros(len(self.mesh_points))
        initialSolution["der_u"] = np.zeros(len(self.mesh_points))

        self.initialSolution = initialSolution


    def createModelFunction(self):

        # States
        u = ca.SX.sym('u') # Curvature Control Freedom
        self.states = DecisionVariables.addState(self.states, u, 'u', 'der_u', 1, (-0.25, 0.25), 0, (0, 0), self.initialSolution["u"])

        Kt = ca.SX.sym('Kt') # Track Curvature (1/m) 
        self.states = DecisionVariables.addState(self.states, Kt, 'Kt', 'der_Kt', 1, (-0.5, 0.5), 2, (0, 0), self.initialSolution["Kt"])

        aTheta = ca.SX.sym('aTheta') # Track Heading Angle (rad)
        self.states = DecisionVariables.addState(self.states, aTheta, 'aTheta', 'der_aTheta', 1, (-np.inf, np.inf), 0, (0,0), self.initialSolution["aTheta"])

        xi = ca.SX.sym('xi') # Track X Position (m)
        self.states = DecisionVariables.addState(self.states, xi, 'xi', 'der_x', 100, (-np.inf, np.inf), 0, (0,0), self.initialSolution["xi"])

        yi = ca.SX.sym('yi') # Track Y Position (m)
        self.states = DecisionVariables.addState(self.states, yi, 'yi', 'der_y', 100, (-np.inf, np.inf), 0, (0,0), self.initialSolution["yi"])

        # Controls
        der_u = ca.SX.sym('der_u') # Curvature Smoothing Factor
        self.controls = DecisionVariables.addControl(self.controls, der_u, 'der_u', 1, (-1, 1), self.initialSolution["der_u"])

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
        rhs[0] = der_u
        rhs[1] = u
        rhs[2] = Kt
        rhs[3] = ca.cos(aTheta)
        rhs[4] = ca.sin(aTheta)

        cost = ((x_ref - xi)/100)**2 + ((y_ref - yi)/100)**2 + (c*u)**2

        # Model Function
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, cost, self.path_constraints.sym, self.auxiliary_outputs.sym],['x', 'u', 'g'], ['rhs', 'cost', 'path_constraints', 'auxiliary_outputs'])