import matplotlib.pyplot as plt
import scienceplots as splt
import pandas as pd
import os

if __name__ == "__main__":

    base_trackmap = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/racetrack-database/racelines/Austin.csv')
    smoothed_trackmap = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne/Austin.csv')
    # e.g. /Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest/1e3.csv

    # Plotting
    plt.style.use('science')
    plt.figure(figsize=(10, 6))
    plt.plot(base_trackmap['# x_m'], base_trackmap['y_m'], label='Base Track')
    plt.plot(smoothed_trackmap['xi'], smoothed_trackmap['yi'], label='Smoothed Track')
    plt.title('Track Map Comparison')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # for label, df in results.items():
    #     plt.plot(df['xi'], df['yi'], label=label)  # Scatter plot for better visualization of points
    # plt.title('Track Layout Optimization Results')
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.legend()
    # plt.grid()
    # plt.axis('equal')  # Ensure equal scaling for x and y axes
    # plt.show()