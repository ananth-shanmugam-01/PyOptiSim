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
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/TestForPR/'

# Sweep Parameters
# sweep_params = {
#     'chassis': {
#         'SCx': [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
#     },
#     }

# sweep_params = {
#     'chassis': {
#         'rAeroBalance': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     },
# }

sweep_params = {
    'chassis': {
        'weightDistribution': [0.3, 0.38, 0.42, 0.46, 0.50, 0.54, 0.6],
    },
}

# sweep_params = {
#     'chassis': {
#         'rollStiffnessDistribution': [0.3, 0.38, 0.42, 0.46, 0.50, 0.54, 0.6],
#     },
# }

# sweep_params = {
#     'tyre': {
#         'lateral_grip_scalar': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
#         'longitudinal_grip_scalar': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
#     },
# }

# sweep_params = {
#     'chassis': {
#         'SCz': np.array(np.linspace(3.8-0.5, 3.8+0.5, 5)),
#         'SCx': np.array(np.linspace(1.2-0.5, 1.2+0.5, 5)),
#     },
# }

# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'gradient-based',}

# Iterate over sweep parameters and their values
for category, params in sweep_params.items():
    for param_name, param_values in params.items():
        for param_value in param_values:

            # Instantiate Model
            modelFun = CarModel()

            # Function update parameters and car data based on overrite functions
            modelFun = loadTrackData(modelFun, 'src/model/Car/component/dataFiles/FSUK_2023_processed.csv')
            endPoint = modelFun.settings['track']['sLap'][-1]
            numIntervals = 200 # Number of Phases

            modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
            modelFun.createMesh(endPoint, numIntervals)
            modelFun.loadCarData()

            # Update the corresponding parameter in modelFun.settings
            if category in modelFun.settings and param_name in modelFun.settings[category]:
                modelFun.settings[category][param_name] = param_value
            else:
                print(f"Warning: {category}.{param_name} not found in modelFun.settings. Skipping update.")

            # Update simulation name to reflect the current parameter value
            sim_name = f"{category}_{param_name}_{param_value}"

            # if sim_name has periods or spaces, replace them with underscores
            sim_name = sim_name.replace('.', '_').replace(' ', '_')

            # Solve the optimization problem for the current parameter set
            print(f"Running simulation for {category}.{param_name} = {param_value}")
            modelFun.createInitialSolution()
            modelFun.createModelFunction()
            optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
            optiProblem.solver('ipopt', p_opts, s_opts)
            sol = optiProblem.solve()

            SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'FSUK_{sim_name}.csv')