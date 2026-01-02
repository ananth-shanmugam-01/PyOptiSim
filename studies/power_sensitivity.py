""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {'30kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_30kW_0001.csv',
            '40kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_40kW_0001.csv',
            '50kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_50kW_0001.csv',
            '60kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_60kW_0001.csv',
            '70kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_70kW_0001.csv',
            '80kW': '/Users/ananthshanmugam/Desktop/SimResults/PMGUK/FSUK_80kW_0001.csv'}

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
power_levels = []
for sim_name, results_df in results_data.items():
    laptime = results_df['t'].iloc[-1]
    laptimes.append(laptime)
    power_level = float(sim_name.replace('kW', ''))
    power_levels.append(power_level)
    plt.plot(power_level, laptime, 'o')
plt.plot(power_levels, laptimes, '-')
plt.xlabel('Max Power (kW)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - Power Sensitivity')
plt.grid(True)
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/PMGUK/', 'FSUK_Power_Sensitivity_Overview.png'), dpi=300)
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
axs[0].set_title('FSUK LapSim Results - Power Sensitivity')

axs[1].set_ylabel('{Ax} (m/s²)')

axs[2].set_ylabel('{Ay} (m/s²)')

axs[3].set_xlabel('sLap (m)')
axs[3].set_ylabel('{P} (kW)')
axs[3].set_ylim([-200, 100])

plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/PMGUK/', 'FSUK_Power_Sensitivity_TimeSeries.png'), dpi=300)
plt.show()