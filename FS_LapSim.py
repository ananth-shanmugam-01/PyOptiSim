#%% Generic Optimal Control Sim

#  Import CasADi
import casadi as ca

# Model Physics
from model.Car.CarModel import CarModel
from model.Car.component.track import loadTrackData

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
          "nlp_scaling_method": 'gradient-based',}

# Instantiate Model
modelFun = CarModel()

# Function update parameters and car data based on overrite functions
modelFun = loadTrackData(modelFun, 'src/model/Car/component/dataFiles/FSUK_2023_processed.csv')
endPoint = modelFun.settings['track']['sLap'][-1]
numIntervals = 200 # Number of Phases

modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
modelFun.createMesh(endPoint, numIntervals)
modelFun.loadCarData()
modelFun.createInitialSolution()
modelFun.createModelFunction()
        
# Create and Solve OCP
optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
optiProblem.solver('ipopt',p_opts,s_opts)
sol = optiProblem.solve() 

# Assigning Values to Dict
SimOut = SimOutputs.createOutputDict(optiProblem, modelFun, Xs, Us, Gs)

SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, 'FSUK_2023_results.csv')