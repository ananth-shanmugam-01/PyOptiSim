import numpy as np
import matplotlib.pyplot as plt

def DebugSim(modelFun, SimOut):

    """ 
    Debug Failed Simulations
    arguments:
        modelFun : Model Function Object
        SimOut   : Simulation Output Object
    Returns:
        plots of states, controls, and parameters
        prints checks for NaNs and Infeasibilities
    
    """
    # Create tiled layout (MATLAB-like) and plot each state into the next tile
    num_states = modelFun.states.num_x
    num_cols = 2
    num_rows = int(np.ceil(num_states / num_cols))

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, constrained_layout=True)

    # Ensure axs is a 2D numpy array indexed as [row, col]
    axs = np.array(axs).reshape(num_rows, num_cols)

    for idx in range(num_states):
        r = idx // num_cols
        c = idx % num_cols
        ax = axs[r, c]
        name = modelFun.states.name[idx]
        ax.plot(SimOut.mesh, SimOut.states[name])
        ax.set_title(f'State: {name}')
        ax.set_xlabel('Mesh [m]')
        ax.set_ylabel(name)

    # Hide any unused tiles
    total_tiles = num_rows * num_cols
    for idx in range(num_states, total_tiles):
        r = idx // num_cols
        c = idx % num_cols
        axs[r, c].set_visible(False)

    plt.show()

    print("Checking for NaNs in Initial Guess:")
    for key, value in modelFun.initialSolution.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"{key}: NaN or Inf detected? {np.any(np.isnan(value)) | np.any(np.isinf(value))}")
        
    print("Checking for NaNs in States:")
    for key, value in SimOut.states.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")

    print("Checking for NaNs in Controls:")
    for key, value in SimOut.controls.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")

    print("Checking for NaNs in Parameters:")
    for key, value in SimOut.parameters.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")

    # Check for infeasibilities in dynamics
    print("Checking for infeasibilities in Dynamics:")
    for key, value in SimOut.der_states.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")

    # Check for infeasibilities in path constraints
    print("Checking for infeasibilities in Path Constraints:")
    for key, value in SimOut.path_constraints.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")

    # Check for Infeasibilities in cost
    print("Checking for infeasibilities in Cost:")
    if np.any(np.isnan(SimOut.cost)) | np.any(np.isinf(SimOut.cost)):
        print(f"Cost: NaN or Inf detected")
    
    # Check for infeasibilities in auxiliary outputs
    print("Checking for infeasibilities in Auxiliary Outputs:")
    for key, value in SimOut.auxiliary_outputs.items():
        if np.any(np.isnan(value)) | np.any(np.isinf(value)):
            print(f"NaN or Inf detected in {key}")
    