#%% Generic Optimal Control Sim

#  Import CasADi
import casadi as ca
import numpy as np

# Model Physics
from model.Car.CarModel import CarModel
from model.Car.component.track import loadTrackData

# Transcription
import src.tools.OptiProblem as OptiProblem

# Post Processing
import src.tools.SimOutputs as SimOutputs

# For Visualisation
import matplotlib.pyplot as plt

# User Settings
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep'

sim_groups = {
    # '0kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': 0
    #     },
    # },
    # '2kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -2e3
    #     },
    # },
    # '4kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -4e3
    #     },
    # },
    # '6kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -6e3
    #     },
    # },
    # '8kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -8e3
    #     },
    # },
    # '10kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -10e3
    #     },
    # },
    # '12kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -12e3
    #     },
    # },
    # '14kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -14e3
    #     },
    # },
    # '20kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -20e3
    #     },
    # },
    # '30kW': {
    #     'powertrain': {
    #         'PMGUKHarvestMax': -30e3
    #     },
    # },
    '50kW': {
        'powertrain': {
            'PMGUKHarvestMax': -50e3
        },
    },
    '70kW': {
        'powertrain': {
            'PMGUKHarvestMax': -70e3
        },
    },
    # '0.38kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.38 * 3.6e6
    #     },
    # },
    # '0.36kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.36 * 3.6e6
    #     },
    # },
    # '0.34kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.34 * 3.6e6
    #     },
    # },
    # '0.32kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.32 * 3.6e6
    #     },
    # },
    # '0.30kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.30 * 3.6e6
    #     }, 
    # },
    # '0.28kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.28 * 3.6e6
    #     },
    # },
    # '0.27kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.27 * 3.6e6
    #     },
    # },
    # '0.26kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.26 * 3.6e6
    #     },
    # },
    # '0.25kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.25 * 3.6e6
    #     },
    # },
    # '0.24kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.24 * 3.6e6
    #     },
    # },
    # '0.23kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.23 * 3.6e6
    #     },
    # },
    # '0.22kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.22 * 3.6e6
    #     },
    # },
    # '0.21kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.21 * 3.6e6
    #     },
    # },
    # '0.20kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.20 * 3.6e6
    #     },
    # },
    # '0.19kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.19 * 3.6e6
    #     },
    # },
    # '0.18kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.18 * 3.6e6
    #     },
    # },
    # '0.17kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.17 * 3.6e6
    #     },
    # },
    # '0.16kWh': {
    #     'powertrain': {
    #         'DeltaSoCLimit': -0.16 * 3.6e6
    #     },
    # },   
}


# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'gradient-based',}

# For each item in sim_groups, we will assign a separate sim and overwwrite the corresponding parameters in the modelFun.settings before solving the optimization problem.
for sim_name, sim_params in sim_groups.items():

    # Instantiate Model
    modelFun = CarModel()

    # Function update parameters and car data based on overrite functions
    modelFun = loadTrackData(modelFun, 'src/model/Car/component/dataFiles/FSUK_2023_processed.csv')
    endPoint = modelFun.settings['track']['sLap'][-1]
    numIntervals = 200 # Number of Phases

    modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
    modelFun.createMesh(endPoint, numIntervals)
    modelFun.loadCarData()

    for category, params in sim_params.items():
        for param_name, param_value in params.items():
            if category in modelFun.settings and param_name in modelFun.settings[category]:
                modelFun.settings[category][param_name] = param_value
                print(f"Updated {category}.{param_name} to {param_value}")
            else:
                print(f"Warning: {category}.{param_name} not found in modelFun.settings. Skipping update.")

    # Solve the optimization problem for the current parameter set
    print(f"Running {sim_name}...")
    modelFun.createInitialSolution()
    modelFun.createModelFunction()
    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
    optiProblem.solver('ipopt', p_opts, s_opts)
    sol = optiProblem.solve()

    print(f"Completed Simulation: {sim_name}, saving results to csv...")
    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'{sim_name}.csv')