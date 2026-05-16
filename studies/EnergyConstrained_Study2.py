""" Simple Script to Load and View Results from LapSim """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    
            '0kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_0kW.csv',
            '2kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_2kW.csv',
            '4kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_4kW.csv',
            '6kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_6kW.csv',
            '8kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_8kW.csv',
            '10kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Endurance_Harvest_10kW.csv',
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

# Scatter plot of Laptime vs Energy Consumed
plt.figure(figsize=(8, 5))

# Line plot of Laptime vs Energy Consumed for all simulations
energy_deployed_list = []
harvest_power_list = []
laptime_list = []
energy_harvested_list = []

for sim_name, results_df in results_data.items():
    harvest_power_list.append(int(sim_name.strip('kW')))

    # Element-wise multiplication of power components and time step, then sum to get total energy
    dt = np.diff(results_df['t'], prepend=results_df['t'].iloc[0])  # Ensure dt matches the length of t
    energy_deployed = np.cumsum(results_df['pmguk_deploy'] * dt) / 3600000 # Convert from J to kWh
    energy_harvested = np.cumsum(results_df['pmguk_harvest'] * dt) / 3600000 # Convert from J to kWh

    energy_deployed_list.append(energy_deployed.iloc[-1])
    energy_harvested_list.append(energy_harvested.iloc[-1])

    laptime_list.append(results_df['t'].iloc[-1])

# plt.plot(harvest_power_list, energy_deployed_list, '-o', color='black')
# plt.xlabel('Harvest Power (kW)')
# plt.ylabel('Energy Deployed (kWh)')
# plt.title('Energy Deployed vs Harvest Power')
# plt.grid(True)
# plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Energy_Deployed_vs_Harvest_Power.png', dpi=300, bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(harvest_power_list, energy_harvested_list, '-o', color='black')
# plt.xlabel('Harvest Power (kW)')
# plt.ylabel('Energy Harvested (kWh)')
# plt.title('Energy Harvested vs Harvest Power')
# plt.grid(True)
# plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Energy_Harvested_vs_Harvest_Power.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot on separate yaxes, but with same yaxis spacing
plt.figure(figsize=(8, 5))
plt.plot(harvest_power_list, energy_deployed_list - np.min(energy_deployed_list), '-o', color='red', label='Energy Deployed')
plt.plot(harvest_power_list, energy_harvested_list - np.max(energy_harvested_list), '-o', color='blue', label='Energy Harvested')
plt.xlabel('Harvest Power (kW)')
plt.ylabel('Energy (kWh)')
plt.title('Energy vs Harvest Power')
plt.grid(True)
plt.legend()
plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Energy_vs_Harvest_Power.png', dpi=300, bbox_inches='tight')
plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(harvest_power_list, laptime_list, '-o', color='black')
# plt.xlabel('Harvest Power (kW)')
# plt.ylabel('Laptime (s)')
# plt.title('Laptime vs Harvest Power')
# plt.grid(True)
# plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_3/Laptime_vs_Harvest_Power.png', dpi=300, bbox_inches='tight')
# plt.show()ß