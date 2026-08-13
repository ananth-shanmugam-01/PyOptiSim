import casadi as ca
import numpy as np

def createOptiProblem(model):

    opti = ca.Opti()
    
    cost = 0
    
    Xs = []
    Us = []
    Gs = []
    
    # Multiply decision variables with their corresponding scales
    Xk = opti.variable( model.states.num_x ) * model.states.scale
    Uk = opti.variable( model.controls.num_u ) * model.controls.scale
    Gk = opti.parameter( model.parameters.num_g )

    # Apply bounds to states and controls
    # States
    for i in range(model.states.num_x):
        opti.subject_to ( opti.bounded(model.states.lb[i] / model.states.scale[i], Xk[i, :] / model.states.scale[i], model.states.ub[i] / model.states.scale[i]) )

    # Controls
    for i in range( model.controls.num_u ):
        opti.subject_to ( opti.bounded(model.controls.lb[i] / model.controls.scale[i], Uk[i, :] / model.controls.scale[i], model.controls.ub[i] / model.controls.scale[i]) )

    Xs = ca.horzcat(Xs, Xk ) 
    Us = ca.horzcat(Us, Uk ) 
    Gs = ca.horzcat(Gs, Gk ) 
    
    for i in range( model.mesh_numIntervals ):
        
        Xc = opti.variable( model.states.num_x, model.collocation_degree ) * model.states.scale
        Uc = opti.variable( model.controls.num_u, model.collocation_degree ) * model.controls.scale
        Gc = opti.parameter( model.parameters.num_g, model.collocation_degree )
        
        # Apply bounds to states and controls
        # States
        for j in range(model.states.num_x):
            opti.subject_to( opti.bounded( model.states.lb[j] / model.states.scale[j], Xc[j, :] / model.states.scale[j], model.states.ub[j] / model.states.scale[j]) )

        # Controls
        for j in range( model.controls.num_u ):
            opti.subject_to( opti.bounded( model.controls.lb[j] / model.controls.scale[j], Uc[j, :] / model.controls.scale[j], model.controls.ub[j] / model.controls.scale[j]) )

        Xs = ca.horzcat(Xs, Xc ) 
        Us = ca.horzcat(Us, Uc ) 
        Gs = ca.horzcat(Gs, Gc ) 
        
        rhs, L, path_constraints, _ = model.modelFunction(Xc, Uc, Gc)

        # If path constraints is not an empty list
        if path_constraints.size(1) > 0:
            for j in range( model.path_constraints.num_path ):
                # Path Constraint Bounds
                opti.subject_to ( opti.bounded( model.path_constraints.lb[j] / model.path_constraints.scale[j], path_constraints[j,:] / model.path_constraints.scale[j], model.path_constraints.ub[j] / model.path_constraints.scale[j]) )
        
        cost = cost + np.matmul( L , model.collocation_B * model.mesh_size )
        
        Z_s = ca.horzcat( Xk , Xc )
        Z_u = ca.horzcat( Uk , Uc )
        
        # Get slope of the interpolating polynomial    
        Pidot = ( 1 / model.mesh_size ) * np.matmul( Z_s , model.collocation_C )
    
        # Scaling the Equality Constraints
        for ii in range(Pidot.shape[0]):
            opti.subject_to( Pidot[ii,:] / model.states.scale[ii] == rhs[ii,:] / model.states.scale[ii] )
        
        Xk_end = np.matmul( Z_s, model.collocation_D )
        Uk_end = np.matmul( Z_u, model.collocation_D )
        
        Xk = opti.variable( model.states.num_x ) * model.states.scale
        Uk = opti.variable( model.controls.num_u ) * model.controls.scale
        Gk = opti.parameter( model.parameters.num_g )

        # Continuity Constraints
        for ii in range(model.states.num_x):
            opti.subject_to( Xk_end[ii] / model.states.scale[ii] == Xk[ii] / model.states.scale[ii] )

        for ii in range(model.controls.num_u):
            opti.subject_to( Uk_end[ii] / model.controls.scale[ii] == Uk[ii] / model.controls.scale[ii] )

        # Apply bounds to states and controls
        # States
        for j in range(model.states.num_x):
            opti.subject_to( opti.bounded( model.states.lb[j] / model.states.scale[j], Xk[j,:] / model.states.scale[j], model.states.ub[j] / model.states.scale[j]) )

        # Controls
        for j in range( model.controls.num_u ):
            opti.subject_to( opti.bounded( model.controls.lb[j] / model.controls.scale[j], Uk[j,:] / model.controls.scale[j], model.controls.ub[j] / model.controls.scale[j]) )

        Xs = ca.horzcat(Xs, Xk ) 
        Us = ca.horzcat(Us, Uk ) 
        Gs = ca.horzcat(Gs, Gk ) 
    
    # Decision Variable Settings
    
    # States
    for i in range(model.states.num_x):

        # Initial Solution
        opti.set_initial( Xs[i,:], model.states.x_init[i] )
        
        # Boundary Conditions
        # BC - 0 - No BC, 1 - Initial Fixed, 2 - Final Fixed, 3 - continuity, 4 - Initial and Terminal Fixed
        if model.states.BC[i] == 4:
            # Initial and Terminal Fixed
            opti.subject_to( Xs[i, 0] / model.states.scale[i] == model.states.BCini[i] / model.states.scale[i])
            opti.subject_to( Xs[i, -1] / model.states.scale[i] == model.states.BCend[i] / model.states.scale[i] )
                            
        elif model.states.BC[i] == 3:
            # Continuity
            opti.subject_to( Xs[i, 0] / model.states.scale[i] == Xs[i, -1] / model.states.scale[i])
        
        elif model.states.BC[i] == 2:
            # Final Value Fixed
            opti.subject_to( Xs[i, -1] / model.states.scale[i] == model.states.BCend[i] / model.states.scale[i] )
        
        elif model.states.BC[i] == 1:
            # Initial Value Fixed
            opti.subject_to( Xs[i, 0] / model.states.scale[i] == model.states.BCini[i] / model.states.scale[i])
        
        elif model.states.BC[i] == 0:
            # No Boundary Conditions
            continue
        
        else:
            # No Boundary Condition
            raise ValueError("Boundary Conditions are Incorrectly Defined")     
       
    
    # Controls
    for i in range( model.controls.num_u ):
        # Initial Solution
        opti.set_initial( Us[i,:], model.controls.u_init[i] )
    
    # Parameters
    for i in range( model.parameters.num_g ):
        opti.set_value( Gs[i,:], model.parameters.value[i] )
    
    # Objective
    opti.minimize( cost )
    
    return opti, Xs, Us, Gs