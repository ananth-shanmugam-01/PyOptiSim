""" Simple Script to Load and View Results from LapSim """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

def calculate_postpro_metrics(results_df):
    metrics = {}

    # Element-wise multiplication of power components and time step, then sum to get total energy
    dt = np.diff(results_df['t'], prepend=results_df['t'].iloc[0])  # Ensure dt matches the length of t
    energy_deployed = np.cumsum(results_df['pmguk_deploy'] * dt) / 3600000 # Convert from J to kWh
    energy_harvested = np.cumsum(results_df['pmguk_harvest'] * dt) / 3600000 # Convert from J to kWh

    metrics['energy_deployed'] = energy_deployed.iloc[-1]
    metrics['energy_harvested'] = energy_harvested.iloc[-1]
    metrics['laptime'] = results_df['t'].iloc[-1]

    return metrics

sim_list = {    
            '0kW Harvest': 
            { 
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.8kWh.csv',
            }
}

# check that files exist
for sim_group_name, sim_group in sim_list.items():
    for sim_name, sim_path in sim_group.items():
        if not os.path.isfile(sim_path):
            raise FileNotFoundError(f"Simulation results file not found: {sim_path}")

# Plot results from each simulation programmatically, based on number of layers inside the dict
results_data = {}
for sim_group_name, sim_group in sim_list.items():
    results_data[sim_group_name] = {}
    for sim_name, sim_path in sim_group.items():
        results_df = pd.read_csv(sim_path)
        results_data[sim_group_name][sim_name] = {'timeseries': results_df}
        # Processing of Metrics
        results_data[sim_group_name][sim_name]['metrics'] = calculate_postpro_metrics(results_df)

plt.style.use(['science', 'grid'])

# Time series plots
fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True, constrained_layout=False)

for sim_group_name, sim_group in results_data.items():
    for sim_name, sim_data in sim_group.items():
        axs[0].plot(sim_data['timeseries']['sLap'], sim_data['timeseries']['u'], label=f'{sim_name} - Laptime: {sim_data["metrics"]["laptime"]:.2f} s')

        axs[1].plot(sim_data['timeseries']['sLap'], sim_data['timeseries']['acc_x'], label=f'{sim_name}')

        axs[2].plot(sim_data['timeseries']['sLap'], sim_data['timeseries']['power_battery']/1e3, label=f'{sim_name}')

        axs[3].plot(sim_data['timeseries']['sLap'], -sim_data['timeseries']['DeltaSoC']/3.6e6, label=f'{sim_name} - Laptime: {sim_data["metrics"]["laptime"]:.2f} s')
        # Set axis labels and titles
axs[0].set_ylabel('Vx (m/s)')
axs[0].set_title('0kW Harvest - Battery Capacity Sweep')

axs[1].set_ylabel('{Ax} (m/s²)')
axs[1].set_ylim([-20, 20])

axs[2].set_ylabel('P_deploy (kW)')
axs[2].set_ylim([-40, 90])
ticks = [-20, 0, 20, 40, 60, 80]
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([f"{t}" for t in ticks])

axs[2].set_ylabel('P_deploy (kW)')
axs[2].set_ylim([-40, 90])
ticks = [-20, 0, 20, 40, 60, 80]
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([f"{t}" for t in ticks])

axs[3].set_xlabel('sLap (m)')
axs[3].set_ylabel('Energy (kWh)')
axs[3].legend(loc='lower left')


plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/LiCo.png', dpi=300, bbox_inches='tight')
plt.show()