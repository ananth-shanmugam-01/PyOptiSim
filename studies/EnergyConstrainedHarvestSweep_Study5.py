""" Simple Script to Load and View Results from LapSim """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {    
            '0kW' : {
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_0kW_5.8kWh.csv',
            },
            '5kW' : {
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.8kWh.csv',
            },
            '10kW' : {
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.8kWh.csv',
            },
            }
# check that files exist
for sim_name, items in sim_list.items():
    for energy_level, energy_path in items.items():
        if not os.path.isfile(energy_path):
            raise FileNotFoundError(f"Simulation results file not found: {energy_path}")

# Plot results from each simulation programmatically
results_data = {sim_name: {} for sim_name in sim_list.keys()}

plt.figure()

for sim_name, items in sim_list.items():
    for energy_level, energy_path in items.items():
        if not os.path.isfile(energy_path):
            raise FileNotFoundError(f"Simulation results file not found: {energy_path}")

        results_df = pd.read_csv(energy_path)
        results_data[sim_name][energy_level] = results_df

    # Plot results for each simulation and energy level
    for energy_level, data in results_data[sim_name].items():
        plt.plot(data['DeltaSoC'].iloc[-1], data['t'].iloc[-1], 'o-', label=f'{sim_name}')
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid()
        plt.legend()


plt.show()