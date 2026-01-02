""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {'0.6': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_0_6.csv',
            '0.8': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_0_8.csv',
            '1.0': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_1_0.csv',
            '1.2': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_1_2.csv',
            '1.4': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_1_4.csv',
            '1.6': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_1_6.csv',
            '1.8': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_1_8.csv',
            '2.0': '/Users/ananthshanmugam/Desktop/SimResults/SCx/FSUK_chassis_SCx_2_0.csv',
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
SCx_levels = []
for sim_name, results_df in results_data.items():
    laptime = results_df['t'].iloc[-1]
    laptimes.append(laptime)
    SCx = float(sim_name.replace('FSUK_chassis_SCx_', '').replace('.csv', ''))
    SCx_levels.append(SCx)
    plt.plot(SCx, laptime, 'o')
plt.plot(SCx_levels, laptimes, '-')
plt.xlabel('SCx (-)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - SCx Sensitivity')
plt.grid(True)
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/SCx/', 'FSUK_SCx_Sensitivity_Overview.png'), dpi=300)
plt.show()

# Time series plots
fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
for sim_name, results_df in results_data.items():
    axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')
    axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')
    axs[2].plot(results_df['sLap'], results_df['acc_y'], label=f'{sim_name}')
    axs[3].plot(results_df['sLap'], results_df['power_wheel']/1e3, label=f'{sim_name}')

axs[0].set_ylabel('Vx (m/s)')
axs[0].legend()
axs[0].set_title('FSUK LapSim Results - SCx Sensitivity')

axs[1].set_ylabel('{Ax} (m/s²)')
axs[1].set_ylim([0, 20])

axs[2].set_ylabel('{Ay} (m/s²)')

axs[3].set_ylabel('{P} (kW)')
axs[3].set_xlabel('sLap (m)')
axs[3].set_ylim([-200, 100])

plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/SCx/', 'FSUK_SCx_Sensitivity_TimeSeries.png'), dpi=300)
plt.show()