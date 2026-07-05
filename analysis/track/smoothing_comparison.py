import matplotlib.pyplot as plt
import scienceplots as splt
import pandas as pd
import os

if __name__ == "__main__":

    base_results_path = '/Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest/'
    # e.g. /Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest/1e3.csv

    # Find file names in this folder
    files = os.listdir(os.path.dirname(base_results_path))

    results = dict()  # Dictionary to hold results for each file
    for file in files:
        if file.endswith('.csv'):
            print(f"Loading results from {file}...")
            results[file] = pd.read_csv(f'{base_results_path}/{file}')

    # Plotting
    plt.style.use('science')
    plt.figure(figsize=(10, 6))
    for label, df in results.items():
        plt.plot(df['sLap'], df['Kt'], label=label)
    plt.title('Track Curvature Optimization Results')
    plt.xlabel('Arc Length (m)')
    plt.ylabel('Curvature (1/m)')
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