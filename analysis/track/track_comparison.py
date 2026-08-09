import matplotlib.pyplot as plt
import scienceplots as splt
import pandas as pd
import os

if __name__ == "__main__":

    fsuk_trackmap = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaStudent/FSUK_2023_processed.csv')
    formulaone_trackmap = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne/Spielberg.csv')
    # e.g. /Users/ananthshanmugam/Desktop/SimResults/TrackMaking/Budapest/1e3.csv

    # Plotting
    plt.style.use('science')
    plt.figure(figsize=(10, 6))
    plt.plot(fsuk_trackmap['xi'], fsuk_trackmap['yi'], label='FSUK Track')
    plt.plot(formulaone_trackmap['xi'], formulaone_trackmap['yi'], label='Formula One Track')
    plt.title('Track Map Comparison')
    plt.xlabel('sLap (m)')
    plt.ylabel('Curvature (1/m)')
    plt.legend()
    plt.grid()
    plt.show()

    # plt.style.use('science')
    # plt.figure(figsize=(10, 6))
    # plt.plot(fsuk_trackmap['sLap'], fsuk_trackmap['theta'], label='FSUK Track')
    # plt.plot(formulaone_trackmap['sLap'], formulaone_trackmap['aTheta'], label='Formula One Track')
    # plt.title('Track Map Comparison')
    # plt.xlabel('sLap (m)')
    # plt.ylabel('Heading Angle (rad)')
    # plt.legend()
    # plt.grid()
    # plt.show()