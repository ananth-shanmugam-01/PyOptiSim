""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt
import os

# Load Results

sim_list = {'0.3': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_3.csv',
            '0.38': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_38.csv',
            '0.42': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_42.csv',
            '0.46': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_46.csv',
            '0.50': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_5.csv',
            '0.54': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_54.csv',
            '0.6': '/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/FSUK_chassis_weightDistribution_0_6.csv',
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
# Sensitivity Plots, plot all sim_list names with their laptimes
plt.figure(figsize=(8, 5))
laptimes = []
rWeightDistribution_levels = []
for sim_name, results_df in results_data.items():
    laptime = results_df['t'].iloc[-1]
    laptimes.append(laptime)
    rWeightDistribution = float(sim_name.replace('FSUK_chassis_weightDistribution_', '').replace('.csv', ''))
    rWeightDistribution_levels.append(rWeightDistribution)
    plt.plot(rWeightDistribution, laptime, 'o')
plt.plot(rWeightDistribution_levels, laptimes, '-')
plt.xlabel('rWeightDistribution (-)')
plt.ylabel('Laptime (s)')
plt.title('FSUK LapSim - rWeightDistribution Sensitivity')
plt.grid(True)
plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/rWeightDistribution/', 'FSUK_rWeightDistribution_Sensitivity_Overview.png'), dpi=300)
plt.show()

# # Time series plots
# fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
# for sim_name, results_df in results_data.items():
#     axs[0].plot(results_df['sLap'], results_df['u'], label=f'{sim_name} - Laptime: {results_df["t"].iloc[-1]:.2f} s')
#     axs[1].plot(results_df['sLap'], results_df['acc_x'], label=f'{sim_name}')
#     axs[2].plot(results_df['sLap'], results_df['acc_y'], label=f'{sim_name}')
#     axs[3].plot(results_df['sLap'], results_df['Fz_fl'], label=f'{sim_name}')
#     axs[3].plot(results_df['sLap'], results_df['Fz_fr'], label=f'{sim_name}')
#     axs[4].plot(results_df['sLap'], results_df['Fz_rl'], label=f'{sim_name}')
#     axs[4].plot(results_df['sLap'], results_df['Fz_rr'], label=f'{sim_name}')

# axs[0].set_ylabel('Vx (m/s)')
# axs[0].legend()
# axs[0].set_title('FSUK LapSim Results - hCoG Sensitivity')

# axs[1].set_ylabel('{Ax} (m/s²)')

# axs[2].set_ylabel('{Ay} (m/s²)')

# axs[3].set_ylabel('{Fz} (N)')

# axs[4].set_xlabel('sLap (m)')
# axs[4].set_ylabel('{Fz} (N)')

# plt.savefig(os.path.join('/Users/ananthshanmugam/Desktop/SimResults/rAeroBalance/', 'FSUK_rAeroBalance_Sensitivity_TimeSeries.png'), dpi=300)
# plt.show()