""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    
            '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/BatteryCapacitySweep_v2/PackCapacity_5_8kWh.csv',
            '5.29kWh' : '/Users/ananthshanmugam/Desktop/SimResults/BatteryCapacitySweep_v2/PackCapacity_5_29kWh.csv',
            '4.84kWh' : '/Users/ananthshanmugam/Desktop/SimResults/BatteryCapacitySweep_v2/PackCapacity_4_84kWh.csv',
            '4.35kWh' : '/Users/ananthshanmugam/Desktop/SimResults/BatteryCapacitySweep_v2/PackCapacity_4_35kWh.csv',
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

# Scatter Plot of track position colored by battery power
fig, ax = plt.subplots(figsize=(10, 6))
# Base Track Position Plot in Black, hide labels for legend
ax.plot(results_data['5.8kWh']['x_ir'], results_data['5.8kWh']['y_ir'], color='black', label='_nolegend_', linewidth=2)
# Scatter of track position with a unique color per simulation
cmap = plt.get_cmap('tab10')
for i, (sim_name, results_df) in enumerate(results_data.items()):
    color = cmap(i % cmap.N)
    bPositiveBatteryPower = results_df['power_battery'] > 1e3  # Only plot points where battery power is positive
    ax.scatter(results_df['x_ir'][bPositiveBatteryPower], results_df['y_ir'][bPositiveBatteryPower], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s', color=color, s=20, alpha=1)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('FSUK LapSim Results - Deployment Strategies')
plt.legend()
# plt.show()

# Time series plots
fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
for sim_name, results_df in results_data.items():
    axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')
    axs[0].axhline(results_df['u'].max(), color='gray', linestyle='--', label='Max Velocity')
    axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')
    axs[2].plot(results_df['sLap'], results_df['acc_y'], label=f'{sim_name}')
    axs[3].plot(results_df['sLap'], results_df['power_battery']/1e3, label=f'{sim_name}')
    axs[4].plot(results_df['sLap'], results_df['DeltaSoC'], label=f'{sim_name} - DeltaSoC')

axs[0].set_ylabel('Vx (m/s)')
axs[0].legend()
axs[0].set_title('FSUK LapSim Results - Battery SoC Delta Sweep')

axs[1].set_ylabel('{Ax} (m/s²)')
axs[1].set_ylim([-20, 20])

axs[2].set_ylabel('{Ay} (m/s²)')
axs[2].set_ylim([-20, 20])

axs[3].set_ylabel('{P} (kW)')
# axs[3].set_ylim([-50, 110])
# axs[3].set_yticks(range(-50, 110, 20))
axs[3].legend()

axs[4].set_xlabel('sLap (m)')
axs[4].set_ylabel('DeltaSoC (J)')

# plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/PMGUK/', 'FSUK_Power_Sensitivity_TimeSeries.png'), dpi=300)
plt.show()