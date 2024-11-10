## Optimal Control Lap Time Simulation
# 20/10/24

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import src.tools.DecisionVariables as DecisionVariables
import src.tools.Mesh as Mesh

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
# addState(states, sym, name, der_name, scale, bounds, BC, BC_Vals, initialSolution):
# BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
velocity = ca.SX.sym('velocity')
velocity_init = 10 * np.ones(len(track.mesh)) 
states = DecisionVariables.addState(states, velocity, 'velocity', 'der_velocity', 10, (0, 100), 4, (10, 0), velocity_init)

mass     = ca.SX.sym('mass')
mass_init = 1 * np.ones(len(track.mesh)) 
states = DecisionVariables.addState(states, mass, 'mass', 'der_mass', 1, (0, 10), 1, (1, 0), mass_init)

# Controls
thrust = ca.SX.sym('thrust')
thrust_init = np.ones(len(track.mesh))
controls = DecisionVariables.addControl(controls, thrust, 'thrust',10, (0, 20), thrust_init)

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

# Test Function
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
    
    rhs, L = f(Xc, Uc, Gc) # Still need to bring in path constraints
    
    cost = cost + np.matmul( L , B * track.meshSize )
    
    # Bring together all the points in this phase [0, 1, 2, 3]
    Z_s = ca.horzcat( Xk , Xc )
    Z_u = ca.horzcat( Uk , Uc )
    
    # Get slope of the interpolating polynomial    
    Pidot = (1 / track.meshSize) * np.matmul( Z_s , C )

    # Scaling the Equality Constraints
    for ii in range(Pidot.shape[0]):
        opti.subject_to( Pidot[ii,:] / states.scale[ii] == rhs[ii,:] / states.scale[ii] )
    
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

#%% Decision Variable Settings

# States
for i in range(states.num_x):
    
    # State Bounds
    opti.subject_to( states.lb[i] / states.scale[i] <= Xs[i,:] / states.scale[i])
    opti.subject_to( Xs[i,:] / states.scale[i] <= states.ub[i] / states.scale[i])
    
    # Initial Solution
    opti.set_initial( Xs[i,:], states.x_init[i] )
    
    # Boundary Conditions
    # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
    if states.BC[i] == 4:
        # Initial and Terminal Fixed
        opti.subject_to( Xs[i, 0] / states.scale[i] == states.BCini[i] / states.scale[i])
        opti.subject_to( Xs[i, -1] / states.scale[i] == states.BCend[i] / states.scale[i] )
                        
    elif states.BC[i] == 3:
        # Continuity
        opti.subject_to( Xs[i, 0] / states.scale[i] == Xs[i, -1] / states.scale[i])
    
    elif states.BC[i] == 2:
        # Final Value Fixed
        opti.subject_to( Xs[i, -1] / states.scale[i] == states.BCend[i] / states.scale[i] )
    
    elif states.BC[i] == 1:
        # Initial Value Fixed
        opti.subject_to( Xs[i, 0] / states.scale[i] == states.BCini[i] / states.scale[i])
        
    else:
        # No Boundary Condition
        raise ValueError("Boundary Conditions are Incorrectly Defined")     
   

# Controls
for i in range(controls.num_u):
    
    # Control Bounds
    opti.subject_to( controls.lb[i] / controls.scale[i] <= Us[i,:] / controls.scale[i] )
    opti.subject_to( Us[i,:] / controls.scale[i] <= controls.ub[i] / controls.scale[i])
    
    # Initial Solution
    opti.set_initial( Us[i,:], controls.u_init[i] )

# Parameters
for i in range(parameters.num_g):
    opti.set_value( Gs[i,:], parameters.value[i] )

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
    
    
    









