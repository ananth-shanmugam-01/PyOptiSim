import matplotlib.pyplot as plt
import scienceplots as splt
import pandas as pd
import os

if __name__ == "__main__":

    track = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/src/model/Car/component/dataFiles/FormulaOne/Spielberg.csv')
    results = pd.read_csv('/Users/ananthshanmugam/Desktop/GitHub/PyOptiSim/tests/Baseline_FormulaOne_Spielberg.csv')

    n_rows = 5

    # Plotting
    plt.style.use('science')

    i = 1
    plt.subplot(n_rows, 1, i)
    plt.plot(results['x_ir'], results['y_ir'], label='Simulation Results')
    plt.plot(track['xi'], track['yi'], label='Track Map')
    plt.title('Track Map Comparison')
    plt.xlabel('X-Position (m)')
    plt.ylabel('Y-Position (m)')
    plt.legend()
    plt.grid()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(results['sLap'], results['u'], label='Simulation Results')
    plt.title('Longitudinal Velocity over Lap')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('Longitudinal Velocity (m/s)')
    plt.legend()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(results['sLap'], results['pmguk_deploy'], label='Deploy')
    plt.plot(results['sLap'], results['pmguk_harvest'], label='Harvest')
    plt.plot(results['sLap'], results['pmguk_deploy']+results['pmguk_harvest'], label='TotalPower')
    plt.title('PMGUK Deployment and Harvesting over Lap')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('Power (W)')
    plt.legend()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(results['sLap'], results['EMGUKHarvest'], label='EMGUKHarvest')
    plt.title('EMGUK Harvesting over Lap')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('EMGUK Harvest [J]')
    plt.legend()

    i += 1
    plt.subplot(n_rows, 1, i)
    plt.plot(results['sLap'], results['DeltaSoC'], label='DeltaSoC')
    plt.title('Delta State of Charge over Lap')
    plt.xlabel('Lap Distance (m)')
    plt.ylabel('DeltaSoC [J]')
    plt.legend()
    plt.show()

# sLap,t,n,xi,u,v,dpsi,x_ir,y_ir,psi,delta,Sxfl,Sxfr,Sxrl,Sxrr,acc_x,acc_y,pmguk_deploy,pmguk_harvest,DeltaSoC,alpha_fl,alpha_fr,alpha_rl,alpha_rr,kappa_fl,kappa_fr,kappa_rl,kappa_rr,Fz_fl,Fz_fr,Fz_rl,Fz_rr,Fx_fl,Fx_fr,Fx_rl,Fx_rr,Fy_fl,Fy_fr,Fy_rl,Fy_rr,power_wheel,power_battery