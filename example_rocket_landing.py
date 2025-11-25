#%% Generic Optimal Control Sim

#  Import CasADi
import casadi as ca

# Model Physics
from src.model.RocketLanding import RocketLanding

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

# Generic Optimal Control Sim

modelFun = RocketLanding.factory()

optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)

# Solve
optiProblem.solver('ipopt',p_opts,s_opts)
sol = optiProblem.solve() 

# Jacobian sparsity pattern
J = ca.jacobian(optiProblem.g, optiProblem.x)
J_val = sol.value(J)
plt.figure()
plt.spy(J_val)
plt.title("Jacobian Sparsity Pattern")

# Hessian sparsity pattern
lam_g = sol.value(optiProblem.lam_g)
L = optiProblem.f + ca.dot(lam_g, optiProblem.g)
H = ca.hessian(L, optiProblem.x)[0]
H_val = sol.value(H)
plt.figure()
plt.spy(H_val)
plt.title("Hessian Sparsity Pattern")

plt.show()

# Assigning Values to Dict
SimOut = SimOutputs.createOutputDict(optiProblem, modelFun, Xs, Us, Gs)

# Plots
RocketLanding.createResultPlots(SimOut)