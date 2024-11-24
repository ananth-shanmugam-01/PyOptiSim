# Rocket Landing Optimal Control Problem
# Lifted from original work in MATLAB
# 13/10/2024

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

#%% Define Model 

# States
x = []
velocity = ca.SX.sym('velocity')
mass     = ca.SX.sym('mass')
x =  ca.vertcat(x, velocity, mass)
num_x = x.size(1)

# Controls
u = []
thrust = ca.SX.sym('thrust')
u = ca.vertcat(u, thrust)
num_u = u.size(1)

# Parameters
p = []
g = ca.SX.sym('g')
p = ca.vertcat(p, g)
num_g = p.size(1)

# Assemble Model 
c = 0.05 # Consumption factor
Sf = 1/velocity

# Model Dynamics
rhs = ca.SX.sym('rhs', num_x)
rhs[0] = Sf * (g - thrust/mass)
rhs[1] = Sf * (-c * thrust)

# Model Penalties
L = Sf

f = ca.Function('f', [x, u, g], [rhs, L],['x', 'u', 'g'], ['rhs', 'L'])
print(f)


[x_dot, cost] = f([8, 1], 4, 1);
print(x_dot)
print(cost)

#%% Define Lagrange Polynomials

d = 3
tau = np.array(ca.collocation_points(d, 'legendre'))

[C, D, B] = ca.collocation_coeff(tau)

#%% Mesh Discretization

s = 50 # [m] Mesh Distance
N = 15 # Number of Phases
h = s/N # mesh size

meshInterval = np.linspace(0, s, N+1)  
numIntervals = N

remesh = np.zeros((N, d+1))

for i in range(len(meshInterval)-1):
    dMesh = h # meshInterval[i+1] - meshInterval[i]
    remesh[i,0] = meshInterval[i]
    for ii in range(d):
        remesh[i, ii+1] = remesh[i,0] + dMesh*tau[ii]

remesh = np.vstack( (np.reshape(remesh, (np.size(remesh),1)), meshInterval[-1]) )

#%% Create Opti Problem
# Decision Variables at each collocation point

opti = ca.Opti()

cost = 0

Xs = []
Us = []
Gs = []

Xk = opti.variable( num_x )
Uk = opti.variable( num_u )
Gk = opti.parameter( num_g )

Xs = ca.horzcat(Xs, Xk ) 
Us = ca.horzcat(Us, Uk ) 
Gs = ca.horzcat(Gs, Gk ) 


for i in range(numIntervals):
    
    Xc = opti.variable( num_x, d )
    Uc = opti.variable( num_u, d )
    Gc = opti.parameter( num_g, d )
    
    Xs = ca.horzcat(Xs, Xc ) 
    Us = ca.horzcat(Us, Uc ) 
    Gs = ca.horzcat(Gs, Gc ) 
    
    rhs, L = f(Xc, Uc, Gc)
    
    cost = cost + np.matmul( L , B * dMesh )
    
    # Bring together all the points in this phase [0, 1, 2, 3]
    Z_s = ca.horzcat( Xk , Xc )
    Z_u = ca.horzcat( Uk , Uc )
    
    # Get slope of the interpolating polynomial
    Pidot = (1 / dMesh) * np.matmul( Z_s , C )
    
    opti.subject_to( Pidot == rhs )
    
    Xk_end = np.matmul( Z_s, D )
    Uk_end = np.matmul( Z_u, D )
    
    Xk = opti.variable( num_x )
    Uk = opti.variable( num_u )
    Gk = opti.parameter( num_g )

    opti.subject_to( Xk_end == Xk )
    opti.subject_to( Uk_end == Uk)
    
    Xs = ca.horzcat(Xs, Xk ) 
    Us = ca.horzcat(Us, Uk ) 
    Gs = ca.horzcat(Gs, Gk ) 

#%% Provide Parameters and Bounds

g_mesh = 9.81 * np.ones(len(remesh))

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
ax1.plot( remesh , x_opt[0,:], '-o')
ax1.set_title('Velocity Trajectory')

ax2.plot(remesh , x_opt[1,:], '-o')
ax2.set_title('Mass State Trajectory')

ax3.plot(remesh , u_opt, '-o')
ax3.set_title('Thrust Control Trajectory')
    
    
    









