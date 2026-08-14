import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

import tools.DecisionVariables as DecisionVariables

from model.BaseModel import BaseModel
from model.Car.component.simpleTyre import simpleTyre as simpleTyre

from maths.smooth import smooth_max, smooth_min, smooth_step

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
        params['chassis']['SCz'] = 4.8 # [-]
        params['chassis']['SCx'] = 1.2 # [-]
        params['chassis']['rAeroBalance'] = 0.44 # [-] Aero Downforce Distribution at Front Axle

        # Powertrain Parameters
        params['powertrain'] = dict()
        params['powertrain']['kDifferential'] = 10.47 # Nms/rad - differential friction coefficient
        params['powertrain']['PICEMax'] = 440e3 # Maximum ICE Power in W
        params['powertrain']['PMGUKDeployMax'] = 350e3 # Maximum Deployment MGUK Power in W
        params['powertrain']['PMGUKHarvestMax'] = -350e3 # Maximum Harvest MGUK Power in W
        params['powertrain']['DeltaSoCLimit'] = 4e6 # Battery Pack Range - SoC Delta limits from Max to Min in J
        params['powertrain']['EMGUKHarvestMax'] = 8e6 # Maximum Harvest MGUK Power at the Battery in W
        params['powertrain']['rBatteryEfficiency'] = 0.95 # Round Trip Battery Efficiency
        params['powertrain']['vCarLimit'] = 150 # m/s - default for unconstrained
        params['powertrain']['isBalancedLap'] = True # If EESS peridiocity is enforced

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

    def runQSSSim(self) -> dict:
        """ 
        Run a very simple QSS sim to get an initial solution for the main car states
        """
        mu_max = 1.4 # Radius of grip circle - maximum friction coefficient
        max_velocity = 80 # m/s - maximum velocity - imposed for simplicity
        max_power = 350e3 # W - maximum power - imposed for simplicity
        car_mass = 700 # kg - car mass - imposed for simplicity

        # Calculate Maximum Velocity Profile based Track Curvature and Car Grip
        track_curvature = self.settings['track']['curv']

        max_velocity_profile = np.minimum(max_velocity, np.sqrt(mu_max * 9.81 / abs(track_curvature + 0.0001)))

        # Calculate Forward Velocity Profile, from the slowest apex
        idx_max_curvature = np.argmax(-max_velocity_profile)

        forward_acceleration_profile = np.zeros_like(max_velocity_profile)
        forward_acceleration_profile[idx_max_curvature] = max_velocity_profile[idx_max_curvature] # Start with the same velocity at the first apex

        for index in range(idx_max_curvature, len(max_velocity_profile) - 1):
            # Calculate the maximum forward acceleration based on the grip circle
            ax_tyre = np.sqrt((mu_max * 9.81) ** 2 - (forward_acceleration_profile[index] ** 2 * abs(track_curvature[index])))
            ax_tractive = max_power / forward_acceleration_profile[index] / car_mass # P = F*v => F = P/v => a = F/m
            ax = min(ax_tyre, ax_tractive)

            forward_acceleration_profile[index+1] = np.sqrt(forward_acceleration_profile[index] ** 2  + 2 * ax * (self.settings['track']['sLap'][index + 1] - self.settings['track']['sLap'][index]))
            forward_acceleration_profile[index+1] = min(forward_acceleration_profile[index+1], max_velocity_profile[index+1]) # Limit to the maximum velocity profile

        # For continuity, first and last points should have the same forward acceleration profile
        forward_acceleration_profile[0] = forward_acceleration_profile[-1]

        for index in range(0, idx_max_curvature):
            # Calculate the maximum forward acceleration based on the grip circle
            ax_tyre = np.sqrt((mu_max * 9.81) ** 2 - (forward_acceleration_profile[index] ** 2 * abs(track_curvature[index])))
            ax_tractive = max_power / forward_acceleration_profile[index] / car_mass # P = F*v => F = P/v => a = F/m
            ax = min(ax_tyre, ax_tractive)

            forward_acceleration_profile[index+1] = np.sqrt(forward_acceleration_profile[index] ** 2  + 2 * ax * (self.settings['track']['sLap'][index + 1] - self.settings['track']['sLap'][index]))
            forward_acceleration_profile[index+1] = min(forward_acceleration_profile[index+1], max_velocity_profile[index+1]) # Limit to the maximum velocity profile

        # Going in reverse for braking profile
        backward_acceleration_profile = np.zeros_like(max_velocity_profile)
        backward_acceleration_profile[-1] = max_velocity_profile[-1] # Start with the same velocity at the last apex
        for index in range(len(max_velocity_profile) - 1, 0, -1):
            # Calculate the maximum forward acceleration based on the grip circle
            ax = np.sqrt((mu_max * 9.81) ** 2 - (forward_acceleration_profile[index] ** 2 * abs(track_curvature[index])))

            backward_acceleration_profile[index-1] = np.sqrt(backward_acceleration_profile[index] ** 2  + 2 * ax * (self.settings['track']['sLap'][index] - self.settings['track']['sLap'][index - 1]))
            backward_acceleration_profile[index-1] = min(backward_acceleration_profile[index-1], max_velocity_profile[index-1]) # Limit to the maximum velocity profile

        final_velocity_profile = np.minimum(forward_acceleration_profile, backward_acceleration_profile)

        # Calculate Auxiliary outputs
        initialSolution = dict()
        initialSolution['sLap'] = self.settings['track']['sLap']
        initialSolution['u'] = final_velocity_profile
        initialSolution['time'] = np.zeros_like(initialSolution['u'])
        for index in range(1, len(initialSolution['u'])):
            initialSolution['time'][index] = initialSolution['time'][index-1] + (initialSolution['sLap'][index] - initialSolution['sLap'][index-1]) / initialSolution['u'][index]
        initialSolution['acc_y'] = initialSolution['u'] ** 2 * track_curvature
        initialSolution['acc_x'] = np.gradient(initialSolution['u'], initialSolution['time'])
        initialSolution['delta'] = self.settings['chassis']['wheelbase'] * track_curvature

        return initialSolution

    def createInitialSolution(self):

        # Interpolate to Main Mesh 
        initialSolution = dict()

        QSSSimResults = self.runQSSSim()

        # Fixed Value for Longitudinal Velocity Initial Solution
        u_init = 10 # m/s
        t_end_init = self.settings['track']['sLap'][-1] / u_init # s

        # Initial Solution for States
        initialSolution["t"] = PchipInterpolator(QSSSimResults['sLap'], QSSSimResults['time'])(self.mesh_points) # s
        initialSolution["n"] = np.zeros(len(self.mesh_points))
        initialSolution["xi"] = np.zeros(len(self.mesh_points))
        initialSolution["u"] = PchipInterpolator(QSSSimResults['sLap'], QSSSimResults['u'])(self.mesh_points) # m/s
        initialSolution["v"] = np.zeros(len(self.mesh_points)) # m/s
        initialSolution["dpsi"] = np.zeros(len(self.mesh_points)) # rad/s
        initialSolution["x_ir"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['xi'])(self.mesh_points) # m - track x-coordinates
        initialSolution["y_ir"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['yi'])(self.mesh_points) # m - track y-coordinates
        initialSolution["psi"] = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['theta'])(self.mesh_points) # rad - track heading angle
        initialSolution["delta"] = PchipInterpolator(QSSSimResults['sLap'], QSSSimResults['delta'])(self.mesh_points) # rad - steering angle
        initialSolution["Sxf"] = np.zeros(len(self.mesh_points)) # front longitudinal slip
        initialSolution["Sxr"] = np.zeros(len(self.mesh_points)) # rear longitudinal slip
        initialSolution["acc_x"] = PchipInterpolator(QSSSimResults['sLap'], QSSSimResults['acc_x'])(self.mesh_points) # m/s^2
        initialSolution["acc_y"] = PchipInterpolator(QSSSimResults['sLap'], QSSSimResults['acc_y'])(self.mesh_points) # m/s^2
        initialSolution["pmguk"] = np.zeros(len(self.mesh_points)) # W
        initialSolution["DeltaSoC"] = np.zeros(len(self.mesh_points)) # J
        initialSolution["rICEThrottle"] = np.zeros(len(self.mesh_points)) # [-]

        # Initial Solution for Controls
        initialSolution["der_delta"] = np.zeros(len(self.mesh_points))
        initialSolution["der_Sxf"] = np.zeros(len(self.mesh_points))
        initialSolution["der_Sxr"] = np.zeros(len(self.mesh_points))
        initialSolution["der_pmguk"] = np.zeros(len(self.mesh_points))
        initialSolution["der_rICEThrottle"] = np.zeros(len(self.mesh_points))

        self.initialSolution = initialSolution

    def createModelFunction(self):

        # States
        # addState(states, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        self.states = DecisionVariables.addState(self.states, 't', 'der_t', 100, (0, 1e3), 1, (0, 0), self.initialSolution["t"] )
        self.states = DecisionVariables.addState(self.states, 'n', 'der_n', 1, (-0.1, 0.1), 3, (0, 0), self.initialSolution["n"] )
        self.states = DecisionVariables.addState(self.states, 'xi', 'der_xi', 1, (np.radians(-4), np.radians(4)), 3, (0, 0), self.initialSolution["xi"] )
        self.states = DecisionVariables.addState(self.states, 'u', 'accx', 10, (1, self.settings['powertrain']['vCarLimit']), 3, (0, 0), self.initialSolution["u"] )
        self.states = DecisionVariables.addState(self.states, 'v', 'accy', 10, (-1e2, 1e2), 3, (0, 0),  self.initialSolution["v"])
        self.states = DecisionVariables.addState(self.states, 'dpsi', 'der_dpsi', 10, (-1e3, 1e3), 3, (0, 0), self.initialSolution["dpsi"])
        self.states = DecisionVariables.addState(self.states, 'x_ir', 'der_x_ir', 1000, (-2000, 2000), 1, (self.settings['track']['xi'][0],  self.settings['track']['xi'][-1]), self.initialSolution["x_ir"])
        self.states = DecisionVariables.addState(self.states, 'y_ir', 'der_y_ir', 1000, (-2000, 2000), 1, (self.settings['track']['yi'][0],  self.settings['track']['yi'][-1]), self.initialSolution["y_ir"])
        self.states = DecisionVariables.addState(self.states, 'psi', 'der_psi', 10, (-10, 10), 1, (self.settings['track']['theta'][0], self.settings['track']['theta'][-1]),  self.initialSolution["psi"])
        self.states = DecisionVariables.addState(self.states, 'delta', 'der_delta', 1, (np.radians(-20), np.radians(20)), 3, (0, 0),  self.initialSolution["delta"])
        self.states = DecisionVariables.addState(self.states, 'Sxfl', 'der_Sxfl', 1, (-0.15, 0), 3, (0, 0),  self.initialSolution["Sxf"])
        self.states = DecisionVariables.addState(self.states, 'Sxfr', 'der_Sxfr', 1, (-0.15, 0), 3, (0, 0),  self.initialSolution["Sxf"])
        self.states = DecisionVariables.addState(self.states, 'Sxrl', 'der_Sxrl', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"])
        self.states = DecisionVariables.addState(self.states, 'Sxrr', 'der_Sxrr', 1, (-0.15, 0.15), 3, (0, 0),  self.initialSolution["Sxr"])
        self.states = DecisionVariables.addState(self.states, 'acc_x', 'der_acc_x', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_x"])
        self.states = DecisionVariables.addState(self.states, 'acc_y', 'der_acc_y', 1e2, (-100, 100), 3, (0, 0),  self.initialSolution["acc_y"])
        self.states = DecisionVariables.addState(self.states, 'pmguk_deploy', 'der_pmguk_deploy', 1e6, (0, self.settings['powertrain']['PMGUKDeployMax']), 3, (0, 0), self.initialSolution["pmguk"])
        self.states = DecisionVariables.addState(self.states, 'pmguk_harvest', 'der_pmguk_harvest', 1e6, (self.settings['powertrain']['PMGUKHarvestMax'], 0), 3, (0, 0), self.initialSolution["pmguk"])
        if self.settings['powertrain']['isBalancedLap']:
            self.states = DecisionVariables.addState(self.states, 'DeltaSoC', 'pmguk_battery', 1e6, (0, self.settings['powertrain']['DeltaSoCLimit']), 3, (0, 0),  self.initialSolution["DeltaSoC"])
        else:
            self.states = DecisionVariables.addState(self.states, 'DeltaSoC', 'pmguk_battery', 1e6, (0, self.settings['powertrain']['DeltaSoCLimit']), 0, (0, 0),  self.initialSolution["DeltaSoC"])
        self.states = DecisionVariables.addState(self.states, 'EMGUKHarvest', 'pmguk_harvest', 1e6, (-1e6, self.settings['powertrain']['EMGUKHarvestMax']), 1, (0, 0),  self.initialSolution["DeltaSoC"])
        self.states = DecisionVariables.addState(self.states, 'rICEThrottle', 'der_rICEThrottle', 10, (0, 1), 3, (0, 0),  self.initialSolution["rICEThrottle"])

        # Controls
        self.controls = DecisionVariables.addControl(self.controls, 'der_delta', 1, (-10, 10), self.initialSolution["der_delta"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_Sxfl', 1, (-10, 10), self.initialSolution["der_Sxf"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_Sxfr', 1, (-10, 10), self.initialSolution["der_Sxf"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_Sxrl', 1, (-10, 10), self.initialSolution["der_Sxr"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_Sxrr', 1, (-10, 10), self.initialSolution["der_Sxr"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_pmguk_deploy', 1e6, (-1e6, 1e6), self.initialSolution["der_pmguk"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_pmguk_harvest', 1e6, (-1e6, 1e6), self.initialSolution["der_pmguk"])
        self.controls = DecisionVariables.addControl(self.controls, 'der_rICEThrottle', 10, (-10, 10), self.initialSolution["der_rICEThrottle"])

        # Parameters
        curv_interp = PchipInterpolator(self.settings['track']['sLap'], self.settings['track']['curv']) (self.mesh_points)
        self.parameters = DecisionVariables.addParameter(self.parameters, 'curv', curv_interp)

        # Vehicle Model
        Fd = -0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCx'] * self.states['u']**2
        Flf = 0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCz'] * self.states['u']**2 * self.settings['chassis']['rAeroBalance']
        Flr = 0.5 * self.settings['constants']['airDensity'] * self.settings['chassis']['SCz'] * self.states['u']**2 * (1 - self.settings['chassis']['rAeroBalance'])

        # Tyre Slip Calculations - looking into using ca.fmax to avoid division by zero, is this smooth enough or continuous enough?
        alpha_fl = -self.states['delta'] + ca.atan2(((self.states['v'] + self.settings['chassis']['frontLeverArm'] * self.states['dpsi'])), self.states['u'] + 0.5 * self.settings['chassis']['halfTrackWidthFront'])
        alpha_fr = -self.states['delta'] + ca.atan2(((self.states['v'] + self.settings['chassis']['frontLeverArm'] * self.states['dpsi'])), self.states['u'] - 0.5 * self.settings['chassis']['halfTrackWidthFront'])
        alpha_rl = ca.atan2(((self.states['v'] - self.settings['chassis']['rearLeverArm'] * self.states['dpsi'])), self.states['u'] + 0.5 * self.settings['chassis']['halfTrackWidthRear'])
        alpha_rr = ca.atan2(((self.states['v'] - self.settings['chassis']['rearLeverArm'] * self.states['dpsi'])), self.states['u'] - 0.5 * self.settings['chassis']['halfTrackWidthRear'])

        # Wheel Longitudinal Slips
        kappa_fl = self.states['Sxfl']
        kappa_fr = self.states['Sxfr']
        kappa_rl = self.states['Sxrl']
        kappa_rr = self.states['Sxrr']

        # Wheel Loads - Static Load + Aero Load + Longitudinal Load Transfer + Lateral Load Transfer
        Fz_fl = (0.5 * self.settings['chassis']['mass'] * self.settings['chassis']['weightDistribution'] * 9.81) + (0.5 * Flf) + (-0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_x'] / self.settings['chassis']['wheelbase']) - (self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_y'] * self.settings['chassis']['rollStiffnessDistribution'] / self.settings['chassis']['trackWidthFront'])
        Fz_fr = (0.5 * self.settings['chassis']['mass'] * self.settings['chassis']['weightDistribution'] * 9.81) + (0.5 * Flf) + (-0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_x'] / self.settings['chassis']['wheelbase']) + (self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_y'] * self.settings['chassis']['rollStiffnessDistribution'] / self.settings['chassis']['trackWidthFront'])
        Fz_rl = (0.5 * self.settings['chassis']['mass'] * (1 - self.settings['chassis']['weightDistribution']) * 9.81) + (0.5 * Flr) + (0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_x'] / self.settings['chassis']['wheelbase']) - (self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_y'] * (1 - self.settings['chassis']['rollStiffnessDistribution']) / self.settings['chassis']['trackWidthRear'])
        Fz_rr = (0.5 * self.settings['chassis']['mass'] * (1 - self.settings['chassis']['weightDistribution']) * 9.81) + (0.5 * Flr) + (0.5 * self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_x'] / self.settings['chassis']['wheelbase']) + (self.settings['chassis']['hCoG'] * self.settings['chassis']['mass'] * self.states['acc_y'] * (1 - self.settings['chassis']['rollStiffnessDistribution']) / self.settings['chassis']['trackWidthRear'])

        # Tyre Forces in Wheel Frame
        Fy_fl, Fx_fl = simpleTyre( kappa_fl, alpha_fl, Fz_fl, self.settings )
        Fy_fr, Fx_fr = simpleTyre( kappa_fr, alpha_fr, Fz_fr, self.settings )
        Fy_rl, Fx_rl = simpleTyre( kappa_rl, alpha_rl, Fz_rl, self.settings )
        Fy_rr, Fx_rr = simpleTyre( kappa_rr, alpha_rr, Fz_rr, self.settings )

        # Tyre Forces in Vehicle Frame
        Fx = ca.cos(self.states['delta']) * (Fx_fl + Fx_fr) - ca.sin(self.states['delta']) * (Fy_fl + Fy_fr) + Fx_rl + Fx_rr + Fd
        Fy = ca.sin(self.states['delta']) * (Fx_fl + Fx_fr) + ca.cos(self.states['delta']) * (Fy_fl + Fy_fr) + Fy_rl + Fy_rr
        Mz = ( self.settings['chassis']['frontLeverArm'] * ( ca.cos(self.states['delta']) * (Fy_fl + Fy_fr) + ca.sin(self.states['delta']) * (Fx_fl + Fx_fr) )
              + self.settings['chassis']['halfTrackWidthFront'] * ( ca.sin(self.states['delta']) * (Fy_fr - Fy_fl) - ca.cos(self.states['delta']) * (Fx_fr - Fx_fl) )
              - self.settings['chassis']['halfTrackWidthRear'] * (Fx_rr - Fx_rl) 
              - self.settings['chassis']['rearLeverArm'] * (Fy_rl + Fy_rr) )
        
        ## Dynamics Scaling Factor
        Sf = (1 - self.states['n'] * self.parameters['curv']) / (self.states['u'] * ca.cos(self.states['xi']) - self.states['v'] * ca.sin(self.states['xi']))
        self.states.derivatives['der_t'] = Sf

        ## Path Dynamics        
        der_acc_x = (-Fx + self.settings['chassis']['mass'] * self.states['acc_x'])/(self.settings['chassis']['mass'] * 0.01)
        der_acc_y = (Fy - self.settings['chassis']['mass'] * self.states['acc_y'])/(self.settings['chassis']['mass'] * 0.01)

        self.states.derivatives['der_n'] = Sf * (self.states['u'] * ca.sin(self.states['xi']) + self.states['v'] * ca.cos(self.states['xi'])) # Path Normal Distance - Dynamics
        self.states.derivatives['der_xi'] = Sf * (self.states['dpsi'] - self.parameters['curv'] / Sf) # Path Heading Angle - Dynamics
        self.states.derivatives['der_psi'] = Sf * self.states['dpsi'] # Vehicle Heading Angle - Dynamics
        self.states.derivatives['der_dpsi'] = Sf * (Mz / self.settings['chassis']['Izz']) # Yaw Rate - Dynamics
        self.states.derivatives['der_x_ir'] = Sf * (self.states['u'] * ca.cos(self.states['psi']) - self.states['v'] * ca.sin(self.states['psi'])) # Global X Position - Dynamics
        self.states.derivatives['der_y_ir'] = Sf * (self.states['u'] * ca.sin(self.states['psi']) + self.states['v'] * ca.cos(self.states['psi'])) # Global Y Position - Dynamics

        ## Chassis Model - Dynamics
        self.states.derivatives['accx'] = Sf * (self.states['dpsi'] * self.states['v'] + self.states['acc_x'])
        self.states.derivatives['accy'] = Sf * (-self.states['dpsi'] * self.states['u'] + self.states['acc_y'])
        self.states.derivatives['der_acc_x'] = Sf * der_acc_x
        self.states.derivatives['der_acc_y'] = Sf * der_acc_y

        # Chassis Model - Path Constraints
        # Non-Negative Wheel Loads
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_fl, 'Fz_fl_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_fr, 'Fz_fr_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_rl, 'Fz_rl_constraint', 1e4, (0, np.inf) )
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, Fz_rr, 'Fz_rr_constraint', 1e4, (0, np.inf) )

        ## Driver Model - Dynamics
        self.states.derivatives['der_delta'] = Sf * self.controls['der_delta']
        self.states.derivatives['der_Sxfl'] = Sf * self.controls['der_Sxfl']
        self.states.derivatives['der_Sxfr'] = Sf * self.controls['der_Sxfr']
        self.states.derivatives['der_Sxrl'] = Sf * self.controls['der_Sxrl']
        self.states.derivatives['der_Sxrr'] = Sf * self.controls['der_Sxrr']

        ## Powertrain Model - Dynamics 

        # For Deployment, bring in FIA rules
        # 1. if PMGUK > 100e3, then der_pmguk_deploy is limited to -100e3, otherwise, derivative is free
        # Use a numerically-stable tanh-based smooth step to avoid exp overflow
        # when the argument is large. The denominator (1000.0) controls the
        # transition width (approx transition over a few thousand Watts).
        # bDerateConstraintActive = smooth_step(self.states['pmguk_deploy'] - 100e3, 5000.0)
        # der_pmguk_deploy = bDerateConstraintActive * smooth_max(-100e3, self.controls['der_pmguk_deploy'], 5e3) + (1 - bDerateConstraintActive) * self.controls['der_pmguk_deploy']

        self.states.derivatives['der_pmguk_deploy'] = Sf * self.controls['der_pmguk_deploy'] # der_pmguk_deploy


        self.states.derivatives['der_pmguk_harvest'] = Sf * self.controls['der_pmguk_harvest']
        self.states.derivatives['pmguk_battery'] = Sf * -1 * (self.states['pmguk_deploy'] * self.settings['powertrain']['rBatteryEfficiency'] + self.states['pmguk_harvest'] / self.settings['powertrain']['rBatteryEfficiency'])
        self.states.derivatives['pmguk_harvest'] = Sf * -1 * self.states['pmguk_harvest']
        self.states.derivatives['der_rICEThrottle'] = Sf * self.controls['der_rICEThrottle']

        # Powertrain Model - Path Constraints
        power_wheel = (Fx - Fd) * self.states['u'] # Cost of drag power
        power_ice = self.states['rICEThrottle'] * self.settings['powertrain']['PICEMax']
        power_mguk = self.states['pmguk_harvest'] / self.settings['powertrain']['rBatteryEfficiency'] + self.states['pmguk_deploy'] * self.settings['powertrain']['rBatteryEfficiency']
        power_constraint = (power_ice + power_mguk) - power_wheel # + self.controls['slack_power'] # Path Constraint
        self.path_constraints = DecisionVariables.addPathConstraint(self.path_constraints, power_constraint, 'power_constraint', 1e6, (0, np.inf) )

        # Stage Cost
        self.penalties = DecisionVariables.penalty()
        self.penalties = DecisionVariables.addPenalty(self.penalties, Sf, 'lap_time')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.01 * self.controls['der_delta']**2, 'steering_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.0005 * self.controls['der_Sxfl']**2, 'front_left_slip_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.0005 * self.controls['der_Sxfr']**2, 'front_right_slip_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.0005 * self.controls['der_Sxrl']**2, 'rear_left_slip_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.0005 * self.controls['der_Sxrr']**2, 'rear_right_slip_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, 0.0005 * self.controls['der_rICEThrottle']**2, 'ice_throttle_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, (1e-10 * self.controls['der_pmguk_deploy'])**2, 'mguk_deploy_rate')
        self.penalties = DecisionVariables.addPenalty(self.penalties, (1e-10 * self.controls['der_pmguk_harvest'])**2, 'mguk_harvest_rate')
        # self.penalties = DecisionVariables.addPenalty(self.penalties, (1e-8 * self.controls['slack_power'])**2, 'slack_power_penalty')

        cost = ca.sum1(self.penalties.sym)

        # Auxiliary Outputs
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fd, 'FDrag')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Flf + Flr, 'FDownforce')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Flf, 'FDownforceF')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Flr, 'FDownforceR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, power_wheel, 'PWheel')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, self.states['pmguk_deploy'] + self.states['pmguk_harvest'], 'PBattery')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_fl, 'FxWheelFL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_fr, 'FxWheelFR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_rl, 'FxWheelRL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fx_rr, 'FxWheelRR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_fl, 'FyWheelFL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_fr, 'FyWheelFR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_rl, 'FyWheelRL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fy_rr, 'FyWheelRR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_fl, 'FWheelLoadFL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_fr, 'FWheelLoadFR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_rl, 'FWheelLoadRL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, Fz_rr, 'FWheelLoadRR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fl * 57.2958, 'aSlipFL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_fr * 57.2958, 'aSlipFR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rl * 57.2958, 'aSlipRL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, alpha_rr * 57.2958, 'aSlipRR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_fl, 'rSlipFL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_fr, 'rSlipFR')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_rl, 'rSlipRL')
        self.auxiliary_outputs = DecisionVariables.addAuxiliaryOutput(self.auxiliary_outputs, kappa_rr, 'rSlipRR')

        # Model Function

        # Assign RHS of the ODEs correctly
        rhs = ca.vertcat(*[
            self.states.derivatives[der_name]
            for der_name in self.states.der_names
        ])

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
    sim_name = 'FormulaOne_Spielberg_ReScale'

    # IPOPT Settings
    p_opts = {}
    s_opts = {
        "max_iter": 800, 
        # "hessian_approximation": 'limited-memory',   # L-BFGS can help if Hessian is noisy
        "mu_strategy": 'adaptive',
        "tol" : 1e-6,
        "acceptable_tol": 1e-4,
        "constr_viol_tol": 1e-3,
        "compl_inf_tol": 1e-3,
        "nlp_scaling_method": 'none',
        }

    # Generic Optimal Control Sim

    modelFun = FormulaOne.factory()

    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

    # Solve
    optiProblem.solver('ipopt',p_opts,s_opts)

    sol = optiProblem.solve()
    SimOut = SimOutputs.createDebugOutputDict(optiProblem, modelFun, Xs, Us, Gs)
    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'{sim_name}.csv')
