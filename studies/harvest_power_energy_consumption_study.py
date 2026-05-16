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
            },
            '5kW Harvest' :
            {
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_5kW_5.8kWh.csv',
            },
            '10kW Harvest' :
            {
                '4.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.0kWh.csv',
                '4.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.4kWh.csv',
                '4.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_4.8kWh.csv',
                '5.0kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.0kWh.csv',
                '5.4kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.4kWh.csv',
                '5.8kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Endurance_Harvest_10kW_5.8kWh.csv',
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

# Scatter plot of Laptime vs Energy Consumed
plt.figure(figsize=(8, 5))

for sim_group_name, sim_group in results_data.items():
    laptime_list = []
    battery_capacity_list = []
    for sim_name, sim_data in sim_group.items():
        laptime_list.append(sim_data['metrics']['laptime'])
        battery_capacity_list.append(float(sim_name.strip('kWh')))
    plt.plot(battery_capacity_list, laptime_list, '-o', label=sim_group_name)

plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Laptime (s)')
plt.title('Laptime vs Battery Capacity')
plt.grid(True)
plt.legend()
plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/Study_5/Laptime_vs_Battery_Capacity.png', dpi=300, bbox_inches='tight')
plt.show()
