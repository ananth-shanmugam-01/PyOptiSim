""" Simple Script to Load and View Results from LapSim """

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots as splt

# Load Results

sim_name = 'FSUK_Baseline_0001'
sim_output_path = '/Users/ananthshanmugam/Desktop/SimResults/'
results_df = pd.read_csv(f'{sim_output_path}/{sim_name}.csv')

# Plot Settings
plt.style.use(['grid'])
fig, axs = plt.subplots(4, 1)
axs[0].plot(results_df['sLap'], results_df['u'], label='Longitudinal Velocity (m/s)', color='black')
axs[0].set_xlabel('sLap (m)')
axs[0].set_ylabel('Velocity (m/s)')
axs[0].legend()

axs[1].plot(results_df['sLap'], results_df['acc_x'], label='Longitudinal Acceleration (m/s²)', color='black')
axs[1].set_xlabel('sLap (m)')
axs[1].set_ylabel('Acceleration (m/s²)')
axs[1].legend()

axs[2].plot(results_df['sLap'], results_df['acc_y'], label='Lateral Acceleration (m/s²)', color='black')
axs[2].set_xlabel('sLap (m)')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].legend()

axs[3].plot(results_df['sLap'], results_df['power_wheel'], label='Power (W)', color='black')
axs[3].set_xlabel('sLap (m)')
axs[3].set_ylabel('Power (W)')
axs[3].legend()

plt.show()