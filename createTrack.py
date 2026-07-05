#%% Generic Optimal Control Sim

#  Import CasADi
import casadi as ca
import numpy as np

# Model Physics
from model.Track.TrackModel import TrackModel

# Transcription
import tools.OptiProblem as OptiProblem

# Post Processing
import tools.SimOutputs as SimOutputs

# For Visualisation
import matplotlib.pyplot as plt

# User Settings
raw_data_fp = '/Users/ananthshanmugam/Desktop/GitHub/racetrack-database/racelines/Budapest.csv'
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest'

# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'gradient-based',}

sim_groups = {
    "1e3":
        {
        "track": {
                "Kt_smoothing_factor": 1e3
            }
        },
    "1e4":
        {
        "track": {
            "Kt_smoothing_factor": 1e4
        },
    },
    "1e5":{
        "track": {
            "Kt_smoothing_factor": 1e5
        },
    },
    "1e6":{
        "track": {
            "Kt_smoothing_factor": 1e6
            }
        },
    } 

# For each item in sim_groups, we will assign a separate sim and overwwrite the corresponding parameters in the modelFun.settings before solving the optimization problem.
for sim_name, sim_params in sim_groups.items():

    # Instantiate Model
    modelFun = TrackModel()

    # Process Raw Track Data
    modelFun.ProcessRawTrackData(raw_data_fp, smoothing_factor=sim_params["track"]["Kt_smoothing_factor"])
    endPoint = modelFun.settings['track']['sLap'][-1]
    numIntervals = 400 # Number of Phases

    modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
    modelFun.createMesh(endPoint, numIntervals)
    modelFun.createInitialSolution(modelFun.settings['track'])
    modelFun.createModelFunction()

    # Solve the optimization problem for the current parameter set
    print(f"Running {sim_name}...")
    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
    optiProblem.solver('ipopt', p_opts, s_opts)
    sol = optiProblem.solve()

    print(f"Completed Simulation: {sim_name}, saving results to csv...")
    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'{sim_name}.csv')