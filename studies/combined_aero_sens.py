""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
plt.style.use(['science', 'grid'])
import os

# Load Results
units = '[-]'
sim_list = {
            "SCz":{
            '3.3': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCz_3_3.csv',
            '3.55': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCz_3_55.csv',
            '3.8': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCz_3_8.csv',
            '4.05': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCz_4_05.csv',
            '4.3': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCz_4_3.csv',
            },
            'SCx':{
            '0.7': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCx_0_7.csv',
            '0.95': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCx_0_95.csv',
            '1.2': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCx_1_2.csv',
            '1.45': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCx_1_45.csv',
            '1.7': '/Users/ananthshanmugam/Desktop/SimResults/Aero/FSUK_chassis_SCx_1_7.csv',
            }
}

central_points = {
    'SCz': 3.8,
    'SCx': 1.2,
}

# First Level Refers to the sim group, second level refers to individual sim name value pairs
sim_group_names = list(sim_list.keys())

# check that files exist
for sim_group, sim_paths in sim_list.items():
    for sim_name, sim_path in sim_paths.items():
        if not os.path.isfile(sim_path):
            raise FileNotFoundError(f"Simulation results file not found: {sim_path}")

# Save results from each simulation, within each group
results_data = {}
for sim_group, sim_paths in sim_list.items():
    for sim_name, sim_path in sim_paths.items():
        results_df = pd.read_csv(sim_path)
        results_data[f'{sim_group}_{sim_name}'] = results_df



# Sensitivity Plots, plot all sim_list names with their laptimes
plt.figure(figsize=(8, 5))

for sim_group in sim_group_names:
    laptimes = []
    sweep_parameters = []
    for sim_name, results_df in results_data.items():
        if sim_name.startswith(sim_group):
            laptime = results_df['t'].iloc[-1]
            laptimes.append(laptime)
            sweep_param = float(sim_name.replace(f'{sim_group}_', '')) - central_points[sim_group]
            sweep_parameters.append(sweep_param)
            plt.plot(sweep_param, laptime, 'o')
    plt.plot(sweep_parameters, laptimes, '-', label=f'{sim_group}')

plt.xlabel(r'$\Delta$ Aero Coefficients (-)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - Aerodynamics Sensitivity')
plt.grid(True)
plt.legend()
plt.xlim([-0.6, 0.6])
plt.xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/Aero/', 'FSUK_Aero_Sensitivity_Overview.png'), dpi=300)
plt.show()
