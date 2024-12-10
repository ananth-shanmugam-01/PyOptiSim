#%% Generic Optimal Control Sim

# Model Physics
import src.model.Track.TrackModel as TrackModel

# Transcription
import src.tools.OptiProblem as OptiProblem

# Post Processing
import src.tools.SimOutputs as SimOutputs

# For Visualisation
import matplotlib.pyplot as plt

# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'none'}

#%% Generic Optimal Control Sim

trackDataFrame, sLap = TrackModel.model.loadTrack("src/model/Track/Catalunya.csv")
modelFun = TrackModel.model.factory(trackDataFrame, sLap)

optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

# Solve
optiProblem.solver('ipopt',p_opts,s_opts)
sol = optiProblem.solve() 

#%% Assigning Values to Dict

SimOut = SimOutputs.createOutputDict(optiProblem, modelFun, Xs, Us, Gs)
