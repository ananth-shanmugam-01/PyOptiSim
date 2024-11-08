## Optimal Control Lap Time Simulation
# 20/10/24

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import tools.DecisionVariables as DecisionVariables
import tools.Mesh as Mesh

# Outline

# 1. Lagrange Polynomials

# 2. Mesh Definition

# 3. Model Definition
# 3a. Model Parameterisation
# 3b. Model Dynamics (States, Controls, Parameters, )

#%% Define Lagrange Polynomials

collocation_degree= 3
tau = np.array(ca.collocation_points(collocation_degree, 'legendre'))

[C, D, B] = ca.collocation_coeff(tau)

#%% Mesh Definition

endPoint = 50 # [m] Mesh Distance
numIntervals = 15 # Number of Phases

track = Mesh.mesh( endPoint, numIntervals, collocation_degree, tau )

#%% Define Decision Variables

states, controls, parameters = DecisionVariables.initialiseDecisionVariables()

#%% Define Model 

# States
velocity = ca.SX.sym('velocity')
velocity_init = np.linspace(0, 10, 12) 
states = DecisionVariables.addState(states, velocity, 'velocity', 'der_velocity', 1, (0, 100), 1, (10, 0), velocity_init)

mass     = ca.SX.sym('mass')
mass_init = np.ones(10) * 1 
states = DecisionVariables.addState(states, mass, 'mass', 'der_mass', 1, (0, 20), 1, (10, 0), mass_init)

# Controls
u = []
thrust = ca.SX.sym('thrust')
thrust_init = np.ones(len(track.mesh))
controls = DecisionVariables.addControl(controls, thrust, 'thrust',1, (0, 20), thrust_init)

# Parameters
g = ca.SX.sym('g')
g_mesh = 9.81 * np.ones(len(track.mesh))
parameters = DecisionVariables.addParameter(parameters, g, 'g', g_mesh)

# Assemble Model 
c = 0.05 # Consumption factor
Sf = 1/velocity

# Model Dynamics
rhs = ca.SX.sym('rhs', states.num_x)
rhs[0] = Sf * (g - thrust/mass)
rhs[1] = Sf * (-c * thrust)

# Model Penalties
L = Sf

f = ca.Function('f', [states.sym, controls.sym, parameters.sym], [rhs, L],['x', 'u', 'g'], ['rhs', 'L'])
print(f)


[x_dot, cost] = f([8, 1], 4, 1);
print(x_dot)
print(cost)

#%% Create Opti Problem
# Decision Variables at each collocation point

opti = ca.Opti()

cost = 0

Xs = []
Us = []
Gs = []

Xk = opti.variable( states.num_x )
Uk = opti.variable( controls.num_u )
Gk = opti.parameter( parameters.num_g )

Xs = ca.horzcat(Xs, Xk ) 
Us = ca.horzcat(Us, Uk ) 
Gs = ca.horzcat(Gs, Gk ) 


for i in range(numIntervals):
    
    Xc = opti.variable( states.num_x, collocation_degree )
    Uc = opti.variable( controls.num_u, collocation_degree )
    Gc = opti.parameter( parameters.num_g, collocation_degree )
    
    Xs = ca.horzcat(Xs, Xc ) 
    Us = ca.horzcat(Us, Uc ) 
    Gs = ca.horzcat(Gs, Gc ) 
    
    rhs, L = f(Xc, Uc, Gc)
    
    cost = cost + np.matmul( L , B * track.meshSize )
    
    # Bring together all the points in this phase [0, 1, 2, 3]
    Z_s = ca.horzcat( Xk , Xc )
    Z_u = ca.horzcat( Uk , Uc )
    
    # Get slope of the interpolating polynomial
    Pidot = (1 / track.meshSize) * np.matmul( Z_s , C )
    
    opti.subject_to( Pidot == rhs )
    
    Xk_end = np.matmul( Z_s, D )
    Uk_end = np.matmul( Z_u, D )
    
    Xk = opti.variable( states.num_x )
    Uk = opti.variable( controls.num_u )
    Gk = opti.parameter( parameters.num_g )

    opti.subject_to( Xk_end == Xk )
    opti.subject_to( Uk_end == Uk)
    
    Xs = ca.horzcat(Xs, Xk ) 
    Us = ca.horzcat(Us, Uk ) 
    Gs = ca.horzcat(Gs, Gk ) 

#%% Provide Parameters and Bounds

g_mesh = 9.81 * np.ones(len(track.mesh))

opti.set_value(Gs, g_mesh)

# Initial Constraints
opti.subject_to(Xs[0,0] == 10)
opti.subject_to(Xs[1,0] == 1)

# Terminal Constraints
opti.subject_to(Xs[0, -1] == 0)

# State Bounds
opti.subject_to( 0 <= Us[0,:] )
opti.subject_to( Us[0,:] <= 20)

opti.subject_to( 0 <= Xs[0,:] )
opti.subject_to( Xs[0,:] <= 99999 )

opti.subject_to( 0 <= Xs[1,:] )
opti.subject_to( Xs[1,:] <= 99999 )

# Initial Solution
opti.set_initial( Us, 1 )
opti.set_initial( Xs[0,:] , 10 )
opti.set_initial( Xs[1,:] , 1)

# Objective
opti.minimize( cost )

# Solve
opti.solver('ipopt')
sol = opti.solve() 

#%%
x_opt = np.array(opti.value(Xs))
u_opt = np.array(opti.value(Us))
g_opt = np.array(opti.value(Gs))

#%%

# Create two subplots and unpack the output array immediately
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot( track.mesh , x_opt[0,:], '-o')
ax1.set_title('Velocity Trajectory')

ax2.plot(track.mesh , x_opt[1,:], '-o')
ax2.set_title('Mass State Trajectory')

ax3.plot(track.mesh , u_opt, '-o')
ax3.set_title('Thrust Control Trajectory')
    
    
    









