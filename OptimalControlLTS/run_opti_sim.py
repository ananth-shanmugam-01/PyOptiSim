#%% Generic Optimal Control Sim

# Model Physics
import src.model.RocketLanding as RocketLanding

# Transcription
import src.tools.OptiProblem as OptiProblem

# Post Processing
import src.tools.SimOutputs as SimOutputs

# For Visualisation
import matplotlib.pyplot as plt

#%% Generic Optimal Control Sim

modelFun = RocketLanding.model.factory()

optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

# Solve
optiProblem.solver('ipopt')
sol = optiProblem.solve() 

#%% Assigning Values to Dict

SimOut = SimOutputs.createOutputDict(optiProblem, modelFun, Xs, Us, Gs)

#%% Plots

RocketLanding.model.createResultPlots(SimOut)