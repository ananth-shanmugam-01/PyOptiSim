""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {'3.0': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_3.csv',
            '3.4': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_3_4.csv',
            '3.8': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_3_8.csv',
            '4.2': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_4_2.csv',
            '4.6': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_4_6.csv',
            '4.8': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_4_8.csv',
            '5.0': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_5_0.csv',
            '5.2': '/Users/ananthshanmugam/Desktop/SimResults/SCz/FSUK_chassis_SCz_5_2.csv',
            }

# check that files exist
for sim_name, sim_path in sim_list.items():
    if not os.path.isfile(sim_path):
        raise FileNotFoundError(f"Simulation results file not found: {sim_path}")

# Plot results from each simulation programmatically
results_data = {}
for sim_name, sim_path in sim_list.items():
    results_df = pd.read_csv(sim_path)
    results_data[sim_name] = results_df

plt.style.use(['science', 'grid'])
# Sensitivity Plots, plot all sim_list names with their laptimes
plt.figure(figsize=(8, 5))
laptimes = []
SCz_levels = []
for sim_name, results_df in results_data.items():
    laptime = results_df['t'].iloc[-1]
    laptimes.append(laptime)
    SCz = float(sim_name.replace('FSUK_chassis_SCz_', '').replace('.csv', ''))
    SCz_levels.append(SCz)
    plt.plot(SCz, laptime, 'o')
plt.plot(SCz_levels, laptimes, '-')
plt.xlabel('SCz (-)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - SCz Sensitivity')
plt.grid(True)
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/SCz/', 'FSUK_SCz_Sensitivity_Overview.png'), dpi=300)
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