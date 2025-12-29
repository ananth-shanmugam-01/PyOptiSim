import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

import src.tools.DecisionVariables as DecisionVariables

from src.model.BaseModel import BaseModel
from src.model.Car.component.simpleTyre import simpleTyre as simpleTyre
from src.model.Car.component.track import loadTrackData, createSimpleTrack

# Transcription
import src.tools.OptiProblem as OptiProblem

# Post Processing
import src.tools.SimOutputs as SimOutputs

# Sim Debugging
from src.tools.DebugSim import DebugSim

class CarModel(BaseModel):

    def loadCarData(self):
        """ Load Car Data from JSON and update model parameters, until then use the values directly"""

        params = dict()

        # Constants
        params['constants'] = dict()
        params['constants']['g'] = 9.81 # m/s^2 gravitational acceleration
        params['constants']['airDensity'] = 1.225 # kg/m^3 air density at sea level

        # Chassis Parameters
        params['chassis'] = dict()
        params['chassis']['mass'] = 260 # fully loaded mass in kg
        params['chassis']['Izz'] = 110 # yaw inertia in kgm^2
        params['chassis']['wheelbase'] = 1.535 # m
        params['chassis']['weightDistribution'] = 0.45 # [-] front axle weight distribution
        params['chassis']['hCoG'] = 0.28 # m
        params['chassis']['rollStiffnessDistribution'] = 0.5 # [-] front axle roll stiffness distribution
        params['chassis']['trackWidthFront'] = 1.21 # m
        params['chassis']['trackWidthRear'] = 1.21 # m
        params['chassis']['rWheel'] = 0.2032 # m, loaded radius is approx 0.196 [m]

        # Aerodynamic Parameters
        params['chassis']['SCz'] = 3.8 # [-]
        params['chassis']['SCx'] = 1.2 # [-]
        params['chassis']['rAeroBalance'] = 0.48 # [-] Aero Downforce Distribution at Front Axle

        # Powertrain Parameters
        params['powertrain'] = dict()
        params['powertrain']['PMGUKDeployMax'] = 80e3 # Maximum Deployment MGUK Power in W
        params['powertrain']['PMGUKHarvestMax'] = 40e3 # Maximum Harvest MGUK Power in W
        params['powertrain']['EESSCapacity'] = 5.8 * 3.6e6 / 22 # Battery Pack Range in J from kWh - 5.8 kWh pack over 22 laps

        # Tyre Parameters
        params['tyre'] = dict()
        params['tyre']['reference_load_1'] = 600
        params['tyre']['reference_load_2'] = 1100
        params['tyre']['peak_mux_reference_load_1'] = 1.53
        params['tyre']['peak_mux_reference_load_2'] = 1.45
        params['tyre']['peak_mux_slip_load_1'] = 0.11
        params['tyre']['peak_mux_slip_load_2'] = 0.10
        params['tyre']['peak_muy_reference_load_1'] = 1.65
        params['tyre']['peak_muy_reference_load_2'] = 1.50
        params['tyre']['peak_muy_slip_load_1'] = np.radians(9) # radians
        params['tyre']['peak_muy_slip_load_2'] = np.radians(8) # radians
        params['tyre']['longitudinal_shape_factor'] = 1.9
        params['tyre']['lateral_shape_factor']  = 1.7 

        # Calculated Parameters
        params['chassis']['frontLeverArm'] = (1- params['chassis']['weightDistribution']) * params['chassis']['wheelbase'] # m
        params['chassis']['rearLeverArm'] = params['chassis']['weightDistribution'] * params['chassis']['wheelbase'] # m
        params['chassis']['halfTrackWidthFront'] = params['chassis']['trackWidthFront'] / 2.0
        params['chassis']['halfTrackWidthRear'] = params['chassis']['trackWidthRear'] / 2.0
        
        self.settings.update(params)

    def createInitialSolution(self):

        # Interpolate to Main Mesh 
        initialSolution = dict()

        # Initial Solution for States
        initialSolution["t"] = np.linspace(0, 100, len(self.mesh_points)) # s
        initialSolution["n"] = np.zeros(len(self.mesh_points))
        initialSolution["xi"] = np.zeros(len(self.mesh_points))
        initialSolution["u"] = 5 * np.ones(len(self.mesh_points)) # m/s
        initialSolution["v"] = np.zeros(len(self.mesh_points)) # m/s
        initialSolution["dpsi"] = np.zeros(len(self.mesh_points)) # rad/s
        initialSolution["x_ir"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['xi'])(self.mesh_points) # m - track x-coordinates
        initialSolution["y_ir"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['yi'])(self.mesh_points) # m - track y-coordinates
        initialSolution["psi"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['theta'])(self.mesh_points) # rad - track heading angle
        initialSolution["delta"] = np.zeros(len(self.mesh_points)) # rad
        initialSolution["Sxf"] = np.zeros(len(self.mesh_points)) # front longitudinal slip
        initialSolution["Sxr"] = np.zeros(len(self.mesh_points)) # rear longitudinal slip
        initialSolution["acc_x"] = np.zeros(len(self.mesh_points)) # m/s^2
        initialSolution["acc_y"] = np.zeros(len(self.mesh_points)) # m/s^2
        initialSolution["pmguk"] = np.zeros(len(self.mesh_points)) # W
        initialSolution["EESS"] = np.zeros(len(self.mesh_points)) # J

        # Initial Solution for Controls
        initialSolution["der_delta"] = np.zeros(len(self.mesh_points))
        initialSolution["der_Sxf"] = np.zeros(len(self.mesh_points))
        initialSolution["der_Sxr"] = np.zeros(len(self.mesh_points))
        initialSolution["der_pmguk"] = np.zeros(len(self.mesh_points))

        self.initialSolution = initialSolution

    def createModelFunction(self):

        # States
        # addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        t = ca.SX.sym('t')     # time (s)
        self.states = DecisionVariables.addState(self.states, t, 't', 'der_t', 100, (0, 1e3), 1, (0, 0), self.initialSolution["t"] )

        n = ca.SX.sym('n')
        self.states = DecisionVariables.addState(self.states, n, 'n', 'der_n', 1, (-0.1, 0.1), 3, (0, 0), self.initialSolution["n"] )

        xi = ca.SX.sym('xi') # Heading angle deviation (rad)
        self.states = DecisionVariables.addState(self.states, xi, 'xi', 'der_xi', 1, (np.radians(-4), np.radians(4)), 3, (0, 0), self.initialSolution["xi"] )

        u = ca.SX.sym('u')         # vehicle fixed x-velocity (m/s)
        self.states = DecisionVariables.addState(self.states, u, 'u', 'accx', 10, (1, 150), 3, (10, 0), self.initialSolution["u"] )

        v = ca.SX.sym('v')         # vehicle fixed y-velocity (m/s)
        self.states = DecisionVariables.addState(self.states, v, 'v', 'accy', 10, (-1e2, 1e2), 3, (0, 0),  self.initialSolution["v"])

        dpsi = ca.SX.sym('dpsi')   # vehicle yaw rate (rad/s)
        self.states = DecisionVariables.addState(self.states, dpsi, 'dpsi', 'der_dpsi', 10, (-1e3, 1e3), 3, (0, 0), self.initialSolution["dpsi"]) 

        x_ir = ca.SX.sym('x_ir')   # x-Position in global coordinates (m)
        self.states = DecisionVariables.addState(self.states, x_ir, 'x_ir', 'der_x_ir', 100, (-20000, 20000), 0, (self.settings['track']['xi'][0],  self.settings['track']['xi'][-1]), self.initialSolution["x_ir"]) 

        y_ir = ca.SX.sym('y_ir')   # y-Position in global coordinates (m)
        self.states = DecisionVariables.addState(self.states, y_ir, 'y_ir', 'der_y_ir', 100, (-20000, 20000), 0, (self.settings['track']['yi'][0],  self.settings['track']['yi'][-1]), self.initialSolution["y_ir"])     
        
        psi = ca.SX.sym('psi')     # yaw angle (rad)
        self.states = DecisionVariables.addState(self.states, psi, 'psi', 'der_psi', 1, (-200, 200), 0, (0, 0),  self.initialSolution["psi"])     

        delta = ca.SX.sym('delta')  # steering angle (rad)
        self.states = DecisionVariables.addState(self.states, delta, 'delta', 'der_delta', 1, (np.radians(-30), np.radians(30)), 3, (0, 0),  self.initialSolution["delta"])     

        Sxfl = ca.SX.sym('Sxfl')      # front left long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxfl, 'Sxfl', 'der_Sxfl', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxf"]) 
        
        Sxfr = ca.SX.sym('Sxfr')      # front right long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxfr, 'Sxfr', 'der_Sxfr', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxf"]) 

        Sxrl = ca.SX.sym('Sxrl')      # rear left long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxrl, 'Sxrl', 'der_Sxrl', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"]) 

        Sxrr = ca.SX.sym('Sxrr')      # rear right long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxrr, 'Sxrr', 'der_Sxrr', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"]) 

        acc_x = ca.SX.sym('acc_x') # longitudinal acceleration (m/s^2)
        self.states = DecisionVariables.addState(self.states, acc_x, 'acc_x', 'der_acc_x', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_x"]) 

        acc_y = ca.SX.sym('acc_y') # lateral acceleration (m/s^2)
        self.states = DecisionVariables.addState(self.states, acc_y, 'acc_y', 'der_acc_y', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_y"]) 

        # pmguk = ca.SX.sym('pmguk') # MGUK Deploy Power at the Wheel (W)
        # self.states = DecisionVariables.addState(self.states, pmguk, 'pmguk', 'der_pmguk', 1e6, (self.settings['powertrain']['PMGUKHarvestMax'], self.settings['powertrain']['PMGUKDeployMax']), 0, (0, 0), self.initialSolution["pmguk"]) 

        # EESS = ca.SX.sym('EESS') # Battery State of Charge (J)
        # self.states = DecisionVariables.addState(self.states, EESS, 'EESS', 'pmguk', 1e6, (0, self.settings['powertrain']['EESSCapacity']), 1, (0, 0),  self.initialSolution["EESS"]) 

        # Controls
        der_delta = ca.SX.sym('der_delta')
        self.controls = DecisionVariables.addControl(self.controls, der_delta, 'der_delta', 1, (-10, 10), self.initialSolution["der_delta"])

        der_Sxfl = ca.SX.sym('der_Sxfl')
        self.controls = DecisionVariables.addControl(self.controls, der_Sxfl, 'der_Sxfl', 1, (-10, 10), self.initialSolution["der_Sxf"])

        der_Sxfr = ca.SX.sym('der_Sxfr')
        self.controls = DecisionVariables.addControl(self.controls, der_Sxfr, 'der_Sxfr', 1, (-10, 10), self.initialSolution["der_Sxf"])

        der_Sxrl = ca.SX.sym('der_Sxrl')
        self.controls = DecisionVariables.addControl(self.controls, der_Sxrl, 'der_Sxrl', 1, (-10, 10), self.initialSolution["der_Sxr"])

        der_Sxrr = ca.SX.sym('der_Sxrr')
        self.controls = DecisionVariables.addControl(self.controls, der_Sxrr, 'der_Sxrr', 1, (-10, 10), self.initialSolution["der_Sxr"])

        # der_pmguk = ca.SX.sym('der_pmguk')
        # self.controls = DecisionVariables.addControl(self.controls, der_pmguk, 'der_pmguk', 1, (-500e3, 500e3), self.initialSolution["der_pmguk"])
        
        # Parameters
        curv = ca.SX.sym('curv')
        curv_interp = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['curv']) (self.mesh_points)
        self.parameters = DecisionVariables.addParameter(self.parameters, curv, 'curv', curv_interp)

        # Vehicle Model
        Fd = -0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCx'] * u**2
        Flf = 0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCz'] * u**2 * self.settings['chassis']['rAeroBalance']
        Flr = 0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCz'] * u**2 * (1 - self.settings['chassis']['rAeroBalance'])

        # Tyre Slip Calculations - looking into using ca.fmax to avoid division by zero, is this smooth enough or continuous enough?
        alpha_fl = -delta + ca.atan2( ( (v + self.settings['chassis']['frontLeverArm'] * dpsi) ) , u + 0.5 * self.settings['chassis']['halfTrackWidthFront'] )
        alpha_fr = -delta + ca.atan2( ( (v + self.settings['chassis']['frontLeverArm'] * dpsi) ) , u - 0.5 * self.settings['chassis']['halfTrackWidthFront'] )
        alpha_rl = ca.atan2( ( (v - self.settings['chassis']['rearLeverArm'] * dpsi) ) , u + 0.5 * self.settings['chassis']['halfTrackWidthRear'] )
        alpha_rr = ca.atan2( ( (v - self.settings['chassis']['rearLeverArm'] * dpsi) ) , u - 0.5 * self.settings['chassis']['halfTrackWidthRear'] )

        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fl, 'alpha_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fr, 'alpha_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rl, 'alpha_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rr, 'alpha_rr')

        # Wheel Longitudinal Slips
        kappa_fl = Sxfl
        kappa_fr = Sxfr
        kappa_rl = Sxrl
        kappa_rr = Sxrr

        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_fl, 'kappa_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_fr, 'kappa_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_rl, 'kappa_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_rr, 'kappa_rr')

        # Wheel Loads - Static Load + Aero Load + Longitudinal Load Transfer + Lateral Load Transfer
        Fz_fl = ( 0.5 * self.settings['chassis']['mass'] * self.settings['chassis']['weightDistribution'] * 9.81 ) + ( 0.5 * Flf ) + (-0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_x / self.settings['chassis']['wheelbase'] ) - ( self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_y * self.settings['chassis']['rollStiffnessDistribution'] / self.settings['chassis']['trackWidthFront'] )
        Fz_fr = ( 0.5 * self.settings['chassis']['mass'] * self.settings['chassis']['weightDistribution'] * 9.81 ) + ( 0.5 * Flf ) + (-0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_x / self.settings['chassis']['wheelbase'] ) + ( self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_y * self.settings['chassis']['rollStiffnessDistribution'] / self.settings['chassis']['trackWidthFront'] )
        Fz_rl = ( 0.5 * self.settings['chassis']['mass'] * (1-self.settings['chassis']['weightDistribution']) * 9.81 ) + ( 0.5 * Flr ) + (0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_x / self.settings['chassis']['wheelbase'] ) - ( self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_y * (1-self.settings['chassis']['rollStiffnessDistribution']) / self.settings['chassis']['trackWidthRear'] )
        Fz_rr = ( 0.5 * self.settings['chassis']['mass'] * (1-self.settings['chassis']['weightDistribution']) * 9.81 ) + ( 0.5 * Flr ) + (0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_x / self.settings['chassis']['wheelbase'] ) + ( self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * acc_y * (1-self.settings['chassis']['rollStiffnessDistribution']) / self.settings['chassis']['trackWidthRear'] )
     
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_fl, 'Fz_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_fr, 'Fz_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_rl, 'Fz_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_rr, 'Fz_rr')

        # Tyre Forces in Wheel Frame
        Fy_fl, Fx_fl = simpleTyre( kappa_fl, alpha_fl, Fz_fl, self.settings )
        Fy_fr, Fx_fr = simpleTyre( kappa_fr, alpha_fr, Fz_fr, self.settings )
        Fy_rl, Fx_rl = simpleTyre( kappa_rl, alpha_rl, Fz_rl, self.settings )
        Fy_rr, Fx_rr = simpleTyre( kappa_rr, alpha_rr, Fz_rr, self.settings )

        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_fl, 'Fx_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_fr, 'Fx_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_rl, 'Fx_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_rr, 'Fx_rr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_fl, 'Fy_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_fr, 'Fy_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_rl, 'Fy_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_rr, 'Fy_rr')

        # Tyre Forces in Vehicle Frame
        Fx = ca.cos(delta) * (Fx_fl + Fx_fr) - ca.sin(delta) * (Fy_fl + Fy_fr) + Fx_rl + Fx_rr + Fd
        Fy = ca.sin(delta) * (Fx_fl + Fx_fr) + ca.cos(delta) * (Fy_fl + Fy_fr) + Fy_rl + Fy_rr  
        Mz = ( self.settings['chassis']['frontLeverArm'] * ( ca.cos(delta) * (Fy_fl + Fy_fr) + ca.sin(delta) * (Fx_fl + Fx_fr) ) 
              + self.settings['chassis']['halfTrackWidthFront'] * ( ca.sin(delta) * (Fy_fr - Fy_fl) - ca.cos(delta) * (Fx_fr - Fx_fl) ) 
              - self.settings['chassis']['halfTrackWidthRear'] * (Fx_rr - Fx_rl) 
              - self.settings['chassis']['rearLeverArm'] * (Fy_rl + Fy_rr) )
        
        # Dynamics Scaling Factor
        Sf = (1 - n*curv)/(u*ca.cos(xi) - v*ca.sin(xi))

        # Dynamics Equations
        der_n = (u*ca.sin(xi) + v*ca.cos(xi))
        der_xi = Sf * dpsi - curv

        der_acc_x = (-Fx + self.settings['chassis']['mass'] * acc_x)/(self.settings['chassis']['mass'] * 0.1)
        der_acc_y = (Fy - self.settings['chassis']['mass'] * acc_y)/(self.settings['chassis']['mass'] * 0.1)

        der_dpsi = Mz / self.settings['chassis']['Izz']
        der_x_ir = ( u * ca.cos(psi) - v * ca.sin(psi) )
        der_y_ir = ( u * ca.sin(psi) + v * ca.cos(psi) )
        der_psi = dpsi
        power_wheel = Fx * u

        # Power at Wheel Constraint
        power_constraint = power_wheel - self.settings['powertrain']['PMGUKDeployMax']
        # Model Path Constraints
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, power_constraint, 'power_constraint', 1e5, (-np.inf, 0) )

        # Model Dynamics
        rhs = ca.SX.sym('rhs', self.states.num_x)
        rhs[0] = Sf
        rhs[1] = Sf * der_n
        rhs[2] = der_xi
        rhs[3] = Sf * (dpsi*v + acc_x)
        rhs[4] = Sf * (-dpsi*u + acc_y)
        rhs[5] = Sf * der_dpsi
        rhs[6] = Sf * der_x_ir
        rhs[7] = Sf * der_y_ir
        rhs[8] = Sf * der_psi
        rhs[9] = Sf * der_delta
        rhs[10] = Sf * der_Sxfl
        rhs[11] = Sf * der_Sxfr
        rhs[12] = Sf * der_Sxrl
        rhs[13] = Sf * der_Sxrr
        rhs[14] = Sf * der_acc_x
        rhs[15] = Sf * der_acc_y
        # rhs[14] = Sf * der_pmguk
        # rhs[15] = Sf * -pmguk

        # Stage Cost
        cost = ( Sf
                + ( 0.01 * der_delta**2 ) 
                + ( 0.0005 * der_Sxfl**2 ) 
                + ( 0.0005 * der_Sxfr**2 )
                + ( 0.0005 * der_Sxrl**2 )
                + ( 0.0005 * der_Sxrr**2 )
                # + ( 1e-9 * der_pmguk**2 )
            )

        # Model Function
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, cost, self.path_constraints.sym, self.auxiliary_outputs.sym],['x', 'u', 'g'], ['rhs', 'cost', 'path_constraints', 'auxiliary_outputs'])

    def factory():
        
        modelFun = CarModel()

        # Function update parameters and car data based on overrite functions
        modelFun = loadTrackData(modelFun, 'src/model/Car/component/dataFiles/FSUK_2023_processed.csv')

        endPoint = modelFun.settings['track']['sLap'][-1]
        numIntervals = 200 # Number of Phases

        modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
        modelFun.createMesh(endPoint, numIntervals)

        modelFun.loadCarData()

        modelFun.createInitialSolution()
        modelFun.createModelFunction()
        
        return modelFun
    
    def createResultPlots(SimOut):
        """ Functionality Not Implemented yet """
        pass

if __name__ == "__main__":

    # IPOPT Settings
    p_opts = {}
    s_opts = {"max_iter": 1000, 
        "tol" : 1e-6,
        "acceptable_tol": 1e-4,
        "constr_viol_tol": 1e-3,
        "compl_inf_tol": 1e-3,
        "nlp_scaling_method": 'gradient-based',}
    # s_opts = {"max_iter": 1000, 
    #         "tol" : 1e-4,
    #         "acceptable_tol": 1e-2,
    #         "constr_viol_tol": 1e-3,
    #         "acceptable_constr_viol_tol": 1e-2,
    #         "compl_inf_tol": 1e-3,
    #         "dual_inf_tol": 1e-1,
    #         "acceptable_dual_inf_tol": 1e2,
    #         "nlp_scaling_method": 'gradient-based',
    #         "mu_strategy": 'adaptive',
    #         "mu_init": 1e-4,
    #         "mu_target": 1e-6,
    #         "mu_min": 1e-6,}

    # Generic Optimal Control Sim

    modelFun = CarModel.factory()

    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

    # Solve
    optiProblem.solver('ipopt',p_opts,s_opts)
    try:
        sol = optiProblem.solve()
        print("Solver succeeded.")
        # Assigning Values to Dict
        SimOut = SimOutputs.createOutputDict(optiProblem, modelFun, Xs, Us, Gs)

        DebugSim(modelFun, SimOut)

    except Exception as e:
        print("Solver failed. Debugging variable values...")
        SimOut = SimOutputs.createDebugOutputDict(optiProblem, modelFun, Xs, Us, Gs)

        DebugSim(modelFun, SimOut)