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

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot( modelFun.mesh_points , SimOut.states["velocity"], '-o')
ax1.set_title('Velocity Trajectory')

ax2.plot( modelFun.mesh_points , SimOut.states["mass"], '-o')
ax2.set_title('Mass State Trajectory')

ax3.plot( modelFun.mesh_points , SimOut.controls["thrust"], '-o')
ax3.set_title('Thrust Control Trajectory')