import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

import tools.DecisionVariables as DecisionVariables

from model.BaseModel import BaseModel
from model.Car.component.simpleTyre import simpleTyre as simpleTyre

# Transcription
import tools.OptiProblem as OptiProblem

# Post Processing
import tools.SimOutputs as SimOutputs

# Sim Debugging
from tools.DebugSim import DebugSim

class FormulaOne(BaseModel):

    def loadCarData(self):
        """ Load Car Data from JSON and update model parameters, until then use the values directly"""

        params = dict()

        # Constants
        params['constants'] = dict()
        params['constants']['g'] = 9.81 # m/s^2 gravitational acceleration
        params['constants']['airDensity'] = 1.225 # kg/m^3 air density at sea level

        # Chassis Parameters
        params['chassis'] = dict()
        params['chassis']['BaseCarMass'] = 710
        params['chassis']['FuelMass'] = 50
        params['chassis']['mass'] = params['chassis']['BaseCarMass'] + params['chassis']['FuelMass'] # fully loaded mass in kg
        params['chassis']['Izz'] = 450 # yaw inertia in kgm^2
        params['chassis']['wheelbase'] = 3.4 # m
        params['chassis']['weightDistribution'] = 0.47 # [-] front axle weight distribution
        params['chassis']['hCoG'] = 0.3 # m
        params['chassis']['rollStiffnessDistribution'] = 0.5 # [-] front axle roll stiffness distribution
        params['chassis']['trackWidthFront'] = 0.73 * 2 # m
        params['chassis']['trackWidthRear'] = 0.73 * 2 # m
        params['chassis']['rWheel'] = 0.33 # m

        # Aerodynamic Parameters
        params['chassis']['SCz'] = 3 # [-]
        params['chassis']['SCx'] = 0.9 # [-]
        params['chassis']['rAeroBalance'] = 0.44 # [-] Aero Downforce Distribution at Front Axle

        # Powertrain Parameters
        params['powertrain'] = dict()
        params['powertrain']['kDifferential'] = 10.47 # Nms/rad - differential friction coefficient
        params['powertrain']['PMGUKDeployMax'] = 120e3 # Maximum Deployment MGUK Power in W
        params['powertrain']['PMGUKHarvestMax'] = -350e3 # Maximum Harvest MGUK Power in W
        params['powertrain']['DeltaSoCLimit'] = 8e6 # Battery Pack Range - SoC Delta limits from Max to Min in J
        params['powertrain']['EMGUKHarvestMax'] = 8e6 # Maximum Harvest MGUK Power at the Battery in W
        params['powertrain']['rBatteryEfficiency'] = 0.95 # Round Trip Battery Efficiency
        params['powertrain']['vCarLimit'] = 150 # m/s - default for unconstrained

        # Tyre Parameters
        params['tyre'] = dict()
        params['tyre']['reference_load_1'] = 2000
        params['tyre']['reference_load_2'] = 6000
        params['tyre']['peak_mux_reference_load_1'] = 1.75
        params['tyre']['peak_mux_reference_load_2'] = 1.40
        params['tyre']['peak_mux_slip_load_1'] = 0.11
        params['tyre']['peak_mux_slip_load_2'] = 0.10
        params['tyre']['peak_muy_reference_load_1'] = 1.80
        params['tyre']['peak_muy_reference_load_2'] = 1.45
        params['tyre']['peak_muy_slip_load_1'] = np.radians(9) # radians
        params['tyre']['peak_muy_slip_load_2'] = np.radians(8) # radians
        params['tyre']['longitudinal_shape_factor'] = 1.9
        params['tyre']['lateral_shape_factor']  = 1.7 
        params['tyre']['lateral_grip_scalar'] = 1
        params['tyre']['longitudinal_grip_scalar'] = 1

        # Calculated Parameters
        params['chassis']['frontLeverArm'] = (1- params['chassis']['weightDistribution']) * params['chassis']['wheelbase'] # m
        params['chassis']['rearLeverArm'] = params['chassis']['weightDistribution'] * params['chassis']['wheelbase'] # m
        params['chassis']['halfTrackWidthFront'] = params['chassis']['trackWidthFront'] / 2.0
        params['chassis']['halfTrackWidthRear'] = params['chassis']['trackWidthRear'] / 2.0
        
        self.settings.update(params)

        return self

    def loadTrackData(self, trackFile: str): 
        """ Load Track Data from JSON and update model parameters, until then use the values directly"""

        trackData = pd.read_csv(trackFile)

        params = dict()
        params['track'] = dict()
        params['track']['sLap'] = trackData['sLap'].to_numpy()
        params['track']['curv'] = trackData['Kt'].to_numpy()
        params['track']['theta'] = trackData['aTheta'].to_numpy()
        params['track']['xi'] = trackData['xi'].to_numpy()
        params['track']['yi'] = trackData['yi'].to_numpy()

        self.settings.update(params)

        return self

    def createInitialSolution(self):

        # Interpolate to Main Mesh 
        initialSolution = dict()

        # Fixed Value for Longitudinal Velocity Initial Solution
        u_init = 10 # m/s
        t_end_init = self.settings['track']['sLap'][-1] / u_init # s

        # Initial Solution for States
        initialSolution["t"] = np.linspace(0, t_end_init, len(self.mesh_points)) # s
        initialSolution["n"] = np.zeros(len(self.mesh_points))
        initialSolution["xi"] = np.zeros(len(self.mesh_points))
        initialSolution["u"] = u_init * np.ones(len(self.mesh_points)) # m/s
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
        initialSolution["DeltaSoC"] = np.zeros(len(self.mesh_points)) # J

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
        self.states = DecisionVariables.addState(self.states, u, 'u', 'accx', 10, (1, self.settings['powertrain']['vCarLimit']), 3, (0, 0), self.initialSolution["u"] )

        v = ca.SX.sym('v')         # vehicle fixed y-velocity (m/s)
        self.states = DecisionVariables.addState(self.states, v, 'v', 'accy', 10, (-1e2, 1e2), 3, (0, 0),  self.initialSolution["v"])

        dpsi = ca.SX.sym('dpsi')   # vehicle yaw rate (rad/s)
        self.states = DecisionVariables.addState(self.states, dpsi, 'dpsi', 'der_dpsi', 10, (-1e3, 1e3), 3, (0, 0), self.initialSolution["dpsi"]) 

        x_ir = ca.SX.sym('x_ir')   # x-Position in global coordinates (m)
        self.states = DecisionVariables.addState(self.states, x_ir, 'x_ir', 'der_x_ir', 1000, (-2000, 2000), 1, (self.settings['track']['xi'][0],  self.settings['track']['xi'][-1]), self.initialSolution["x_ir"]) 

        y_ir = ca.SX.sym('y_ir')   # y-Position in global coordinates (m)
        self.states = DecisionVariables.addState(self.states, y_ir, 'y_ir', 'der_y_ir', 1000, (-2000, 2000), 1, (self.settings['track']['yi'][0],  self.settings['track']['yi'][-1]), self.initialSolution["y_ir"])     

        psi = ca.SX.sym('psi')     # yaw angle (rad)
        self.states = DecisionVariables.addState(self.states, psi, 'psi', 'der_psi', 10, (-10, 10), 1, (self.settings['track']['theta'][0], self.settings['track']['theta'][-1]),  self.initialSolution["psi"])     

        delta = ca.SX.sym('delta')  # steering angle (rad)
        self.states = DecisionVariables.addState(self.states, delta, 'delta', 'der_delta', 1, (np.radians(-20), np.radians(20)), 3, (0, 0),  self.initialSolution["delta"])     

        Sxfl = ca.SX.sym('Sxfl')      # front left long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxfl, 'Sxfl', 'der_Sxfl', 1, (-0.15, 0), 3, (0, 0),  self.initialSolution["Sxf"]) 
        
        Sxfr = ca.SX.sym('Sxfr')      # front right long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxfr, 'Sxfr', 'der_Sxfr', 1, (-0.15, 0), 3, (0, 0),  self.initialSolution["Sxf"]) 

        Sxrl = ca.SX.sym('Sxrl')      # rear left long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxrl, 'Sxrl', 'der_Sxrl', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"]) 

        Sxrr = ca.SX.sym('Sxrr')      # rear right long. slip (-)
        self.states = DecisionVariables.addState(self.states, Sxrr, 'Sxrr', 'der_Sxrr', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"]) 

        acc_x = ca.SX.sym('acc_x') # longitudinal acceleration (m/s^2)
        self.states = DecisionVariables.addState(self.states, acc_x, 'acc_x', 'der_acc_x', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_x"]) 

        acc_y = ca.SX.sym('acc_y') # lateral acceleration (m/s^2)
        self.states = DecisionVariables.addState(self.states, acc_y, 'acc_y', 'der_acc_y', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_y"]) 

        pmguk_deploy = ca.SX.sym('pmguk_deploy') # MGUK Deploy Power at the Wheel (W)
        self.states = DecisionVariables.addState(self.states, pmguk_deploy, 'pmguk_deploy', 'der_pmguk_deploy', 1e6, (0, self.settings['powertrain']['PMGUKDeployMax']), 3, (0, 0), self.initialSolution["pmguk"]) 

        pmguk_harvest = ca.SX.sym('pmguk_harvest') # MGUK Harvest Power at the Wheel (W)
        self.states = DecisionVariables.addState(self.states, pmguk_harvest, 'pmguk_harvest', 'der_pmguk_harvest', 1e6, (self.settings['powertrain']['PMGUKHarvestMax'], 0), 3, (0, 0), self.initialSolution["pmguk"]) 

        DeltaSoC = ca.SX.sym('DeltaSoC') # Battery State of Charge Delta from Start of Lap (J) - Free Boundary Conditions, as long as it is within the prescribed window
        self.states = DecisionVariables.addState(self.states, DeltaSoC, 'DeltaSoC', 'pmguk_battery', 1e6, (0, self.settings['powertrain']['DeltaSoCLimit']), 0, (0, 0),  self.initialSolution["DeltaSoC"]) 

        # EMGUKHarvest = ca.SX.sym('EMGUKHarvest') # MGUK Harvest Energy at the Battery (J) - Starts at Zero
        # self.states = DecisionVariables.addState(self.states, EMGUKHarvest, 'EMGUKHarvest', 'pmguk_harvest', 1e6, (-1e6, self.settings['powertrain']['EMGUKHarvestMax']), 1, (0, 0),  self.initialSolution["DeltaSoC"]) 

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

        der_pmguk_deploy = ca.SX.sym('der_pmguk_deploy')
        self.controls = DecisionVariables.addControl(self.controls, der_pmguk_deploy, 'der_pmguk_deploy', 1e6, (-1e6, 1e6), self.initialSolution["der_pmguk"])

        der_pmguk_harvest = ca.SX.sym('der_pmguk_harvest')
        self.controls = DecisionVariables.addControl(self.controls, der_pmguk_harvest, 'der_pmguk_harvest', 1e6, (-1e6, 1e6), self.initialSolution["der_pmguk"])

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

        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fl * 57.2958, 'alpha_fl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fr * 57.2958, 'alpha_fr')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rl * 57.2958, 'alpha_rl')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rr * 57.2958, 'alpha_rr')

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

        der_acc_x = (-Fx + self.settings['chassis']['mass'] * acc_x)/(self.settings['chassis']['mass'] * 0.01)
        der_acc_y = (Fy - self.settings['chassis']['mass'] * acc_y)/(self.settings['chassis']['mass'] * 0.01)

        der_dpsi = Mz / self.settings['chassis']['Izz']
        der_x_ir = ( u * ca.cos(psi) - v * ca.sin(psi) )
        der_y_ir = ( u * ca.sin(psi) + v * ca.cos(psi) )
        der_psi = dpsi
        power_wheel = (Fx - Fd) * u # Cost of drag power

        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, power_wheel, 'power_wheel')

        # Power at Wheel Constraint
        power_constraint = (pmguk_harvest / self.settings['powertrain']['rBatteryEfficiency'] + pmguk_deploy * self.settings['powertrain']['rBatteryEfficiency']) - power_wheel # Path Constraint
        
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, pmguk_deploy + pmguk_harvest, 'power_battery')

        # Model Path Constraints
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, power_constraint, 'power_constraint', 1e5, (0, np.inf) )

        # Non-Negative Wheel Loads
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_fl, 'Fz_fl_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_fr, 'Fz_fr_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_rl, 'Fz_rl_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_rr, 'Fz_rr_constraint', 1e4, (0, np.inf) )

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
        rhs[16] = Sf * der_pmguk_deploy
        rhs[17] = Sf * der_pmguk_harvest
        rhs[18] = Sf * -1 * (pmguk_deploy * self.settings['powertrain']['rBatteryEfficiency'] + pmguk_harvest / self.settings['powertrain']['rBatteryEfficiency']) # Deploy results in battery depletion, harvest results in battery charge
        # rhs[19] = Sf * -1 * pmguk_harvest # Harvest results in battery charge

        # Stage Cost
        cost = ( Sf
                + ( 0.01 * der_delta**2 ) 
                + ( 0.0005 * der_Sxfl**2 ) 
                + ( 0.0005 * der_Sxfr**2 )
                + ( 0.0005 * der_Sxrl**2 )
                + ( 0.0005 * der_Sxrr**2 )
                + ( 1e-14 * der_pmguk_deploy)**2
                + ( 1e-14 * der_pmguk_harvest)**2
                # + ( 1e-18 * (pmguk_deploy * pmguk_harvest)**2 )
            )

        # Model Function
        self.modelFunction = ca.Function('f', [self.states.sym, self.controls.sym, self.parameters.sym], [rhs, cost, self.path_constraints.sym, self.auxiliary_outputs.sym],['x', 'u', 'g'], ['rhs', 'cost', 'path_constraints', 'auxiliary_outputs'])

    def factory():

        modelFun = FormulaOne()

        # Function update parameters and car data based on overrite functions
        modelFun = modelFun.loadTrackData('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne/Spielberg.csv')

        endPoint = modelFun.settings['track']['sLap'][-1]
        numIntervals = 300 # Number of Phases

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

    # Simulation Settings
    sim_output_path = '/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/tests'
    sim_name = 'Baseline_FormulaOne_Spielberg'

    # IPOPT Settings
    p_opts = {}
    s_opts = {"max_iter": 200, 
        "tol" : 1e-6,
        "acceptable_tol": 1e-4,
        "constr_viol_tol": 1e-3,
        "compl_inf_tol": 1e-3,
        "nlp_scaling_method": 'gradient-based',}

    # Generic Optimal Control Sim

    modelFun = FormulaOne.factory()

    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

    # Solve
    optiProblem.solver('ipopt',p_opts,s_opts)

    sol = optiProblem.solve()
    SimOut = SimOutputs.createDebugOutputDict(optiProblem, modelFun, Xs, Us, Gs)
    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'{sim_name}.csv')