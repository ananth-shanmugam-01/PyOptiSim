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

#%% Plots

# plt.plot( SimOut.mesh, modelFun.initialSolution["curv"])
# plt.plot( SimOut.mesh, SimOut.states["curv"])
# plt.xlabel("sLap [m]")
# plt.ylabel("Curvature [1/m]")
# plt.title("Track Curvature")
# plt.grid()
# plt.legend(['Initial Solution', 'Optimal Solution'])
# plt.show()        

plt.plot( SimOut.mesh, SimOut.controls["u"])
plt.xlabel("sLap [m]")
plt.ylabel("heading [rad]")
plt.title("Track Heading")
plt.grid()
plt.legend(['Initial Solution', 'Optimal Solution'])
plt.show()     

plt.plot( SimOut.mesh, modelFun.initialSolution["theta"])
plt.plot( SimOut.mesh, SimOut.states["theta"])
plt.xlabel("sLap [m]")
plt.ylabel("heading [rad]")
plt.title("Track Heading")
plt.grid()
plt.legend(['Initial Solution', 'Optimal Solution'])
plt.show()     

plt.plot( modelFun.initialSolution["x"], modelFun.initialSolution["y"])
plt.plot( SimOut.states["xi"], SimOut.states["yi"])
plt.xlabel("x_coordinate [m]")
plt.ylabel("y_coordinate [m]")
plt.title("Track Coordinates")
plt.grid()
plt.legend(['Initial Solution', 'Optimal Solution'])
plt.show()       