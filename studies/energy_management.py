""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    '7.8kWh Limit' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/FSUK_FSUK_EnergyManagement_7_8kWh.csv',
                '5.8kWh Limit' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/FSUK_FSUK_EnergyManagement_5_8kWh.csv',
                '3.8kWh Limit' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/FSUK_FSUK_EnergyManagement_3_8kWh.csv',
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
# Time series plots
fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
for sim_name, results_df in results_data.items():
    axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')
    axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')
    axs[2].plot(results_df['sLap'], results_df['acc_y'], label=f'{sim_name}')
    axs[3].plot(results_df['sLap'], results_df['pmguk']/1e3, label=f'{sim_name}')
    axs[4].plot(results_df['sLap'], results_df['DeltaSoC'], label=f'{sim_name} - DeltaSoC')

axs[0].set_ylabel('Vx (m/s)')
axs[0].legend()
axs[0].set_title('FSUK LapSim Results - Battery SoC Delta Sweep')

axs[1].set_ylabel('{Ax} (m/s²)')
axs[1].set_ylim([-20, 20])

axs[2].set_ylabel('{Ay} (m/s²)')
axs[2].set_ylim([-20, 20])

axs[3].set_ylabel('{P} (W)')
axs[3].set_ylim([-200, 100])

axs[4].set_xlabel('sLap (m)')
axs[4].set_ylabel('DeltaSoC (J)')

# plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/PMGUK/', 'FSUK_Power_Sensitivity_TimeSeries.png'), dpi=300)
plt.show()