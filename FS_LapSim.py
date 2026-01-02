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

# User Settings
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/rAeroBalance_v2/'

# Sweep Parameters
sweep_params = {
    'chassis': {
        'SCx': [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    },
    }

sweep_params = {
    'chassis': {
        'rAeroBalance': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
}
# params['chassis'] = dict()
# params['chassis']['mass'] = 260 # fully loaded mass in kg
# params['chassis']['Izz'] = 110 # yaw inertia in kgm^2
# params['chassis']['wheelbase'] = 1.535 # m
# params['chassis']['weightDistribution'] = 0.45 # [-] front axle weight distribution
# params['chassis']['hCoG'] = 0.28 # m
# params['chassis']['rollStiffnessDistribution'] = 0.5 # [-] front axle roll stiffness distribution
# params['chassis']['trackWidthFront'] = 1.21 # m
# params['chassis']['trackWidthRear'] = 1.21 # m
# params['chassis']['rWheel'] = 0.2032 # m, loaded radius is approx 0.196 [m]

# # Aerodynamic Parameters
# params['chassis']['SCz'] = 3.8 # [-]
# params['chassis']['SCx'] = 1.2 # [-]
# params['chassis']['rAeroBalance'] = 0.48 # [-] Aero Downforce Distribution at Front Axle

# # Powertrain Parameters
# params['powertrain'] = dict()
# params['powertrain']['PMGUKDeployMax'] = 80e3 # Maximum Deployment MGUK Power in W
# params['powertrain']['PMGUKHarvestMax'] = 40e3 # Maximum Harvest MGUK Power in W
# params['powertrain']['EESSCapacity'] = 5.8 * 3.6e6 / 22 # Battery Pack Range in J from kWh - 5.8 kWh pack over 22 laps


# IPOPT Settings
p_opts = {}
s_opts = {"max_iter": 1000, 
          "tol" : 1e-6,
          "acceptable_tol": 1e-4,
          "constr_viol_tol": 1e-3,
          "compl_inf_tol": 1e-3,
          "nlp_scaling_method": 'gradient-based',}
# Iterate over sweep parameters and their values
for category, params in sweep_params.items():
    for param_name, param_values in params.items():
        for param_value in param_values:

            # Instantiate Model
            modelFun = CarModel()

            # Function update parameters and car data based on overrite functions
            modelFun = loadTrackData(modelFun, 'src/model/Car/component/dataFiles/FSUK_2023_processed.csv')
            endPoint = modelFun.settings['track']['sLap'][-1]
            numIntervals = 200 # Number of Phases

            modelFun.createLagrangeCoefficients(3, 'legendre') # collocation degree and strategy
            modelFun.createMesh(endPoint, numIntervals)
            modelFun.loadCarData()

            # Update the corresponding parameter in modelFun.settings
            if category in modelFun.settings and param_name in modelFun.settings[category]:
                modelFun.settings[category][param_name] = param_value
            else:
                print(f"Warning: {category}.{param_name} not found in modelFun.settings. Skipping update.")

            # Update simulation name to reflect the current parameter value
            sim_name = f"{category}_{param_name}_{param_value}"

            # if sim_name has periods or spaces, replace them with underscores
            sim_name = sim_name.replace('.', '_').replace(' ', '_')

            # Solve the optimization problem for the current parameter set
            print(f"Running simulation for {category}.{param_name} = {param_value}")
            modelFun.createInitialSolution()
            modelFun.createModelFunction()
            optiProblem, Xs, Us, Gs = OptiProblem.createOptiProblem(modelFun)
            optiProblem.solver('ipopt', p_opts, s_opts)
            sol = optiProblem.solve()

            SimOutputs.createResultsCSV(optiProblem, modelFun, Xs, Us, Gs, sim_output_path, f'FSUK_{sim_name}.csv')