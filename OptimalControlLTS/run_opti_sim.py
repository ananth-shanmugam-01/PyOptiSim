# Generic Optimal Control Sim

import src.model.RocketLanding as RocketLanding

import src.tools.OptiProblem as OptiProblem
import numpy as np

import matplotlib.pyplot as plt


modelFun = RocketLanding.model.factory()

optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

# Solve
optiProblem.solver('ipopt')
sol = optiProblem.solve() 

#%%
x_opt = np.array(optiProblem.value(Xs))
u_opt = np.array(optiProblem.value(Us))
g_opt = np.array(optiProblem.value(Gs))

#%%

# Create two subplots and unpack the output array immediately
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot( modelFun.mesh_points , x_opt[0,:], '-o')
ax1.set_title('Velocity Trajectory')

ax2.plot( modelFun.mesh_points , x_opt[1,:], '-o')
ax2.set_title('Mass State Trajectory')

ax3.plot( modelFun.mesh_points , u_opt, '-o')
ax3.set_title('Thrust Control Trajectory')