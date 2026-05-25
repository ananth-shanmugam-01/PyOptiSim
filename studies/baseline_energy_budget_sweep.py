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
            'Optimal Deployment': 
            { 
                '0.38kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.38kWh.csv',
                # '0.36kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.36kWh.csv',
                # '0.34kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.34kWh.csv',
                # '0.32kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.32kWh.csv',
                # '0.30kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.30kWh.csv',
                # '0.28kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.28kWh.csv',
                # '0.27kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.27kWh.csv',
                '0.26kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.26kWh.csv',
                # '0.25kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.25kWh.csv',
                # '0.24kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.24kWh.csv',
                # '0.23kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.23kWh.csv',
                # '0.22kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.22kWh.csv',
                # '0.21kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.21kWh.csv',
                # '0.20kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.20kWh.csv',
                # '0.19kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.19kWh.csv',
                # '0.18kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.18kWh.csv',
                # '0.17kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.17kWh.csv',
                # '0.16kWh' : '/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/0.16kWh.csv',
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
    energy_budget_list = []
    for sim_name, sim_data in sim_group.items():
        laptime_list.append(sim_data['metrics']['laptime'])
        energy_budget_list.append(float(sim_name.strip('kWh')))
    plt.plot(energy_budget_list, laptime_list, '-o', label=sim_group_name)

plt.xlabel('Energy Budget (kWh)')
plt.ylabel('Laptime (s)')
plt.title('Laptime vs Energy Budget')
plt.grid(True)
plt.legend()
plt.savefig('/Users/ananthshanmugam/Desktop/SimResults/EnergyManagement/EnergyBudgetSweep/Laptime_vs_Energy_Budget.png', dpi=300, bbox_inches='tight')
plt.show()
