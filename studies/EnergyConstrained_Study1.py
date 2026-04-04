""" Simple Script to Load and View Results from LapSim 
    - Compare Autocross and Endurance Baseline
    - Compare the effect of harvest limits on endurance lap performance
"""

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    
            'Autocross' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Autocross.csv',
            'Endurance Baseline' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Endurance_Baseline.csv',
            # 'Endurance 0kWh Harvest' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Endurance_0kWh_Harvest.csv',
            # 'Endurance 20kWh Harvest' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Endurance_20kWh_Harvest.csv',
            # 'Endurance 30kWh Harvest' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Endurance_30kWh_Harvest.csv',
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
fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True, constrained_layout=False)
for sim_name, results_df in results_data.items():
    axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')

    axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')

    axs[2].plot(results_df['sLap'], results_df['power_battery']/1e3, label=f'{sim_name}')

    axs[3].plot(results_df['sLap'], -results_df['DeltaSoC']/3.6e6, label=f'{sim_name}')
    max_energy = (-results_df['DeltaSoC']/3.6e6).max()
    axs[3].axhline(max_energy, color='red', linestyle='--')
    axs[3].text(
        100,  # x-coordinate (end of the line)
        max_energy-0.05,            # y-coordinate (slightly above the line)
        f"{max_energy:.2f} kWh",     # Text to display
        color='red',                 # Text color
        fontsize=10,                 # Font size
        ha='right'                   # Align text to the right
    )

# Set axis labels and titles
axs[0].set_ylabel('Vx (m/s)')
axs[0].set_title('FSUK LapSim Results - Autocross vs Endurance')
axs[0].legend()

axs[1].set_ylabel('{Ax} (m/s²)')
axs[1].set_ylim([-20, 20])

axs[2].set_ylabel('P_deploy (kW)')
axs[2].set_ylim([-40, 90])
ticks = [-20, 0, 20, 40, 60, 80]
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([f"{t}" for t in ticks])

axs[3].set_xlabel('sLap (m)')
axs[3].set_ylabel('Energy (kWh)')

plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_1/Autocross_vs_Endurance_baseline.png', dpi=300, bbox_inches='tight')
plt.show()