import matplotlib.pyplot as plt
import scienceplots as splt
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
import os

def loadCarData() -> dict:
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

    return params

def loadTrackData(trackFile: str) -> dict:
    """ Load Track Data from JSON and update model parameters, until then use the values directly"""

    trackData = pd.read_csv(trackFile)

    params = dict()
    params['track'] = dict()
    params['track']['sLap'] = trackData['sLap'].to_numpy()
    params['track']['curv'] = trackData['Kt'].to_numpy()
    params['track']['theta'] = trackData['aTheta'].to_numpy()
    params['track']['xi'] = trackData['xi'].to_numpy()
    params['track']['yi'] = trackData['yi'].to_numpy()

    return params

if __name__ == "__main__":

    # Calculate a Simple Velocity Profile using fixed grip and power values
    referenceResults = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/tests/Baseline_FormulaOne_Spielberg.csv')
    trackFile = '/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne/Spielberg.csv'
    trackData = loadTrackData(trackFile)

    mu_max = 1.4 # Radius of grip circle - maximum friction coefficient
    max_velocity = 80 # m/s - maximum velocity - imposed for simplicity
    max_power = 350e3 # W - maximum power - imposed for simplicity
    car_mass = 700 # kg - car mass - imposed for simplicity

    # Calculate Maximum Velocity Profile based Track Curvature and Car Grip
    track_curvature = trackData['track']['curv']

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

        forward_acceleration_profile[index+1] = np.sqrt(forward_acceleration_profile[index] ** 2  + 2 * ax * (trackData['track']['sLap'][index + 1] - trackData['track']['sLap'][index]))
        forward_acceleration_profile[index+1] = min(forward_acceleration_profile[index+1], max_velocity_profile[index+1]) # Limit to the maximum velocity profile

    # For continuity, first and last points should have the same forward acceleration profile
    forward_acceleration_profile[0] = forward_acceleration_profile[-1]

    for index in range(0, idx_max_curvature):
        # Calculate the maximum forward acceleration based on the grip circle
        ax_tyre = np.sqrt((mu_max * 9.81) ** 2 - (forward_acceleration_profile[index] ** 2 * abs(track_curvature[index])))
        ax_tractive = max_power / forward_acceleration_profile[index] / car_mass # P = F*v => F = P/v => a = F/m
        ax = min(ax_tyre, ax_tractive)

        forward_acceleration_profile[index+1] = np.sqrt(forward_acceleration_profile[index] ** 2  + 2 * ax * (trackData['track']['sLap'][index + 1] - trackData['track']['sLap'][index]))
        forward_acceleration_profile[index+1] = min(forward_acceleration_profile[index+1], max_velocity_profile[index+1]) # Limit to the maximum velocity profile

    # Going in reverse for braking profile
    backward_acceleration_profile = np.zeros_like(max_velocity_profile)
    backward_acceleration_profile[-1] = max_velocity_profile[-1] # Start with the same velocity at the last apex
    for index in range(len(max_velocity_profile) - 1, 0, -1):
        # Calculate the maximum forward acceleration based on the grip circle
        ax = np.sqrt((mu_max * 9.81) ** 2 - (forward_acceleration_profile[index] ** 2 * abs(track_curvature[index])))

        backward_acceleration_profile[index-1] = np.sqrt(backward_acceleration_profile[index] ** 2  + 2 * ax * (trackData['track']['sLap'][index] - trackData['track']['sLap'][index - 1]))
        backward_acceleration_profile[index-1] = min(backward_acceleration_profile[index-1], max_velocity_profile[index-1]) # Limit to the maximum velocity profile

    final_velocity_profile = np.minimum(forward_acceleration_profile, backward_acceleration_profile)

    # Calculate Auxiliary outputs
    sLap = trackData['track']['sLap']
    time = np.zeros_like(final_velocity_profile)
    for index in range(1, len(final_velocity_profile)):
        time[index] = time[index-1] + (sLap[index] - sLap[index-1]) / final_velocity_profile[index]
    acc_y = final_velocity_profile ** 2 * track_curvature
    acc_x = np.gradient(final_velocity_profile, time)
    delta = 3.4 * track_curvature 

    # Plotting
    plt.style.use('science')

    n_rows = 5

    i = 1
    plt.subplot(n_rows, 1, i)
    plt.plot(referenceResults['sLap'], referenceResults['u'], label='Reference Velocity Profile')
    plt.plot(trackData['track']['sLap'], max_velocity_profile, label='Max Velocity Profile')
    plt.plot(trackData['track']['sLap'], forward_acceleration_profile, label='Forward Velocity Profile')
    plt.plot(trackData['track']['sLap'], backward_acceleration_profile, label='Backward Velocity Profile')
    plt.plot(trackData['track']['sLap'], final_velocity_profile, label='Final Velocity Profile')
    plt.title('Maximum Velocity Profile based on Track Curvature and Car Grip')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(referenceResults['sLap'], referenceResults['t'], label='Reference')
    plt.plot(sLap, time, label='QSS')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(referenceResults['sLap'], referenceResults['acc_y'], label='Reference Lateral Acceleration Profile')
    plt.plot(sLap, acc_y, label='Lateral Acceleration Profile')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('Lateral Acceleration (m/s²)')
    plt.legend()
    plt.grid()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(referenceResults['sLap'], referenceResults['acc_x'], label='Reference Longitudinal Acceleration Profile')
    plt.plot(sLap, acc_x, label='Longitudinal Acceleration Profile')
    plt.ylabel('Longitudinal Acceleration (m/s²)')
    plt.legend()
    plt.grid()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(referenceResults['sLap'], referenceResults['delta'], label='LapSim Steering Angle Profile')
    plt.plot(sLap, delta, label='Kinematic Steering Angle Profile')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('Steering Angle (degrees)')
    plt.legend()
    plt.grid()
    plt.show()
