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
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/'

sim_groups = {
    'Endurance_Harvest_0kW_5.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -5.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_0kW_5.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -5.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_0kW_5.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -5.0 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_0kW_4.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -4.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_0kW_4.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -4.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_0kW_4.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': 0,
            'DeltaSoCLimit': -4.0 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_5.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -5.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_5.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -5.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_5.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -5.0 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_4.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -4.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_4.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -4.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_5kW_4.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -5e3,
            'DeltaSoCLimit': -4.0 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_5.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -5.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_5.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -5.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_5.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -5.0 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_4.8kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -4.8 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_4.4kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -4.4 * 3.6e6 / 22
        },
    },
    'Endurance_Harvest_10kW_4.0kWh': {
        'powertrain': {
            'PMGUKHarvestMax': -10e3,
            'DeltaSoCLimit': -4.0 * 3.6e6 / 22
        },
    },
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