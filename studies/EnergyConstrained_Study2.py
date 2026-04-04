""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    
            '0.1kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_1kWh.csv',
            '0.15kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_15kWh.csv',
            '0.20kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_20kWh.csv',
            '0.25kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_25kWh.csv',
            '0.30kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_30kWh.csv',
            '0.35kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_35kWh.csv',
            '0.40kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Endurance_0_40kWh.csv',
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
energy_consumed_list = []
laptime_list = []
for sim_name, results_df in results_data.items():
    energy_consumed = (-results_df['DeltaSoC']/3.6e6).max()  # Convert from J to kWh
    energy_consumed_list.append(energy_consumed)
    laptime_list.append(results_df['t'].iloc[-1])

plt.plot(energy_consumed_list, laptime_list, '-o', color='black')

plt.xlabel('Energy Consumed (kWh)')
plt.ylabel('Laptime (s)')
plt.title('Laptime vs Energy Consumed')
plt.grid(True)
plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_2/Laptime_vs_Energy_Consumed.png', dpi=300, bbox_inches='tight')
plt.show()