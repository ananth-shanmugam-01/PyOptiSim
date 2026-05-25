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
            'Harvest Power Sweep': 
            { 
                '0kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/0kW.csv',
                '2kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/2kW.csv',
                '4kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/4kW.csv',
                '6kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/6kW.csv',
                '8kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/8kW.csv',
                '10kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/10kW.csv',
                '12kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/12kW.csv',
                '14kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/14kW.csv',
                '20kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/20kW.csv',
                '30kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/30kW.csv',
                '50kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/50kW.csv',
                '70kW' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/70kW.csv',
            },
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
    harvest_power_list = []
    for sim_name, sim_data in sim_group.items():
        laptime_list.append(sim_data['metrics']['laptime'])
        harvest_power_list.append(float(sim_name.strip('kW')))
    plt.plot(harvest_power_list, laptime_list, '-o', label=sim_group_name)
plt.vlines(10, *plt.ylim(), colors='red', linestyles='--', label='Baseline (10 kW Harvest Power)')
plt.xlabel('Harvest Power (kW)')
plt.ylabel('Laptime (s)')
plt.title('Laptime vs Harvest Power')
plt.grid(True)
plt.legend()
plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/HarvestPowerSweep/Laptime_vs_Harvest_Power.png', dpi=300, bbox_inches='tight')
plt.show()