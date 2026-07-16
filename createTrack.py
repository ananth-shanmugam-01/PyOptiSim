#%% Generic Optimal Control Sim

#  Import CasADi
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots as splt

# Model Physics
from model.Track.TrackModel import TrackModel

# Transcription
import tools.OptiProblem as OptiProblem

# Post Processing
import tools.SimOutputs as SimOutputs

# User Settings
raw_data_base_fp = '/Users/ananthshanmugam/Desktop/GitHub/racetrack-database/racelines'
tracklist_fp = {
    "Austin": f"{raw_data_base_fp}/Austin.csv",
    "Budapest": f"{raw_data_base_fp}/Budapest.csv",
    "Catalunya": f"{raw_data_base_fp}/Catalunya.csv",
    "Hockenheim": f"{raw_data_base_fp}/Hockenheim.csv",
    "Melbourne": f"{raw_data_base_fp}/Melbourne.csv",
    "MexicoCity": f"{raw_data_base_fp}/MexicoCity.csv",
    "Montreal": f"{raw_data_base_fp}/Montreal.csv",
    "Monza": f"{raw_data_base_fp}/Monza.csv",
    "Sakhir": f"{raw_data_base_fp}/Sakhir.csv",
    "SaoPaulo": f"{raw_data_base_fp}/SaoPaulo.csv",
    "Sepang": f"{raw_data_base_fp}/Sepang.csv",
    "Shanghai": f"{raw_data_base_fp}/Shanghai.csv",
    "Silverstone": f"{raw_data_base_fp}/Silverstone.csv",
    "Spa": f"{raw_data_base_fp}/Spa.csv",
    "Spielberg": f"{raw_data_base_fp}/Spielberg.csv",
    "Suzuka": f"{raw_data_base_fp}/Suzuka.csv",
    "YasMarina": f"{raw_data_base_fp}/YasMarina.csv",
    "Zandvoort": f"{raw_data_base_fp}/Zandvoort.csv",
}

output_base_fp = '/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne'

# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'gradient-based',}

# For each in the tracklist, create a track

# For each item in sim_groups, we will assign a separate sim and overwwrite the corresponding parameters in the modelFun.settings before solving the optimization problem.
for track_name, track_fp in tracklist_fp.items():

    # Instantiate Model
    modelFun = TrackModel()

    # Process Raw Track Data
    modelFun.ProcessRawTrackData(os.path.join(raw_data_base_fp, track_fp))
    endPoint = modelFun.settings['track']['sLap'][-1]
    numIntervals = 4000 # Number of Phases

    modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
    modelFun.createMesh(endPoint, numIntervals)
    modelFun.createInitialSolution(modelFun.settings['track'])
    modelFun.createModelFunction()

    # Solve the optimization problem for the current parameter set
    print(f"Running {track_name}...")
    optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
    optiProblem.solver('ipopt', p_opts, s_opts)
    sol = optiProblem.solve()

    print(f"Completed Simulation: {track_name}, saving results to csv...")

    save_fp = os.path.join(output_base_fp, f'{track_name}.csv')
    SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, output_base_fp, save_fp)

    # Plot results for sanity checking
    results = pd.read_csv(save_fp)

    plt.style.use('science')
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(results['sLap'], results['Kt'])
    plt.xlabel('Arc Length (m)')
    plt.ylabel('Curvature (1/m)')
    plt.grid('both')

    plt.subplot(1,2,2)
    plt.plot(results['xi'], results['yi'])
    plt.xlabel('X Position (m)')
    plt.ylabel('y Position (m)')
    plt.axis('equal')
    plt.grid('both')
    plt.savefig(os.path.join(output_base_fp, f'{track_name}.png'))