""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
plt.style.use(['science', 'grid'])
import os

# Load Results
units = '[-]'
sim_list = {
            'lateral_grip_scalar':{
            '0.6': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_0_6.csv',
            '0.7': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_0_7.csv',
            '0.8': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_0_8.csv',
            '0.9': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_0_9.csv',
            '1.0': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_1_0.csv',
            '1.1': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_1_1.csv',
            '1.2': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_1_2.csv',
            '1.3': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_1_3.csv',
            '1.4': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_lateral_grip_scalar_1_4.csv',
            },
            'longitudinal_grip_scalar':{    
            '0.6': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_0_6.csv',
            '0.7': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_0_7.csv',
            '0.8': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_0_8.csv',
            '0.9': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_0_9.csv',
            '1.0': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_1_0.csv',
            '1.1': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_1_1.csv',
            '1.2': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_1_2.csv',
            '1.3': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_1_3.csv',
            '1.4': '/Users/ananthshanmugam/Desktop/SimResults/GripScalar/FSUK_tyre_longitudinal_grip_scalar_1_4.csv',
            },
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
    grip_levels = []
    for sim_name, results_df in results_data.items():
        if sim_name.startswith(sim_group):
            laptime = results_df['t'].iloc[-1]
            laptimes.append(laptime)
            grip_level = float(sim_name.replace(f'{sim_group}_', ''))
            grip_levels.append(grip_level)
            plt.plot(grip_level, laptime, 'o')
    plt.plot(grip_levels, laptimes, '-', label=f'{sim_group}')

plt.xlabel('Grip Scalar (-)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - Grip Scalar Sensitivity')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/GripScalar/', 'FSUK_GripScalar_Sensitivity_Overview.png'), dpi=300)
plt.show()


# # Time series plots
# fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
# for sim_name, results_df in results_data.items():
#     axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')
#     axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')
#     axs[2].plot(results_df['sLap'], results_df['acc_y'], label=f'{sim_name}')
#     axs[3].plot(results_df['sLap'], results_df['Fz_fl'], label=f'{sim_name}')
#     axs[3].plot(results_df['sLap'], results_df['Fz_fr'], label=f'{sim_name}')
#     axs[4].plot(results_df['sLap'], results_df['Fz_rl'], label=f'{sim_name}')
#     axs[4].plot(results_df['sLap'], results_df['Fz_rr'], label=f'{sim_name}')

# axs[0].set_ylabel('Vx (m/s)')
# axs[0].legend()
# axs[0].set_title('FSUK LapSim Results - hCoG Sensitivity')

# axs[1].set_ylabel('{Ax} (m/s²)')

# axs[2].set_ylabel('{Ay} (m/s²)')

# axs[3].set_ylabel('{Fz} (N)')

# axs[4].set_xlabel('sLap (m)')
# axs[4].set_ylabel('{Fz} (N)')

# plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/rAeroBalance/', 'FSUK_rAeroBalance_Sensitivity_TimeSeries.png'), dpi=300)
# plt.show()