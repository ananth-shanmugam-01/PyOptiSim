import numpy as np
import casadi as ca

# Plotting
import matplotlib.pyplot as plt

def tyreParameters():
    params = dict()
    params['tyre'] = dict()
    params['tyre']['reference_load_1'] = 600
    params['tyre']['reference_load_2'] = 1100
    params['tyre']['peak_mux_reference_load_1'] = 1.53
    params['tyre']['peak_mux_reference_load_2'] = 1.45
    params['tyre']['peak_mux_slip_load_1'] = 0.11
    params['tyre']['peak_mux_slip_load_2'] = 0.10
    params['tyre']['peak_muy_reference_load_1'] = 1.65
    params['tyre']['peak_muy_reference_load_2'] = 1.50
    params['tyre']['peak_muy_slip_load_1'] = np.radians(9) # radians
    params['tyre']['peak_muy_slip_load_2'] = np.radians(8) # radians
    params['tyre']['longitudinal_shape_factor'] = 1.9
    params['tyre']['lateral_shape_factor']  = 1.7
    return params


def simpleTyre(kappa: float, alpha: float, Fz: float, settings: dict):

    # From Perantoni & Limebeer (simplified combined-slip friction model)
    Fz1 = settings['tyre']['reference_load_1']  # e.g. 2000
    Fz2 = settings['tyre']['reference_load_2']  # e.g. 6000
    mux_max_1 = settings['tyre']['peak_mux_reference_load_1']  # e.g. 1.75
    mux_max_2 = settings['tyre']['peak_mux_reference_load_2']  # e.g. 1.40
    kappa_1 = settings['tyre']['peak_mux_slip_load_1']  # e.g. 0.11
    kappa_2 = settings['tyre']['peak_mux_slip_load_2']  # e.g. 0.10
    muy_max_1 = settings['tyre']['peak_muy_reference_load_1']  # e.g. 1.80
    muy_max_2 = settings['tyre']['peak_muy_reference_load_2']  # e.g. 1.45
    alpha_1 = settings['tyre']['peak_muy_slip_load_1']  # e.g. 9 (units consistent with input)
    alpha_2 = settings['tyre']['peak_muy_slip_load_2']  # e.g. 8
    Qx = settings['tyre']['longitudinal_shape_factor']  # e.g. 1.9
    Qy = settings['tyre']['lateral_shape_factor']       # corrected key

    # Equations
    mux_max = mux_max_1 + (Fz - Fz1) * (mux_max_2 - mux_max_1)/(Fz2 - Fz1)
    muy_max = muy_max_1 + (Fz - Fz1) * (muy_max_2 - muy_max_1)/(Fz2 - Fz1)

    kappa_max = kappa_1 + (Fz - Fz1) * (kappa_2 - kappa_1)/(Fz2 - Fz1)
    alpha_max = alpha_1 + (Fz - Fz1) * (alpha_2 - alpha_1)/(Fz2 - Fz1)

    kappa_norm = kappa/(kappa_max)
    alpha_norm = alpha/(alpha_max)

    # Combined Slip Coefficient

    rho = ca.sqrt( (alpha_norm**2) + (kappa_norm**2) + 0.001 )

    Sx = ca.pi/(2*ca.atan(Qx))
    Sy = ca.pi/(2*ca.atan(Qy))

    mux = mux_max * ca.sin(Qx * ca.atan(Sx * rho))
    muy = muy_max * ca.sin(Qy * ca.atan(Sy * rho))

    Fx = mux * Fz * kappa_norm/(rho)
    Fy = -muy * Fz * alpha_norm/(rho)

    return Fy, Fx

if __name__ == "__main__":

    settings = tyreParameters()

    Fz_sweep = [300, 600, 900, 1200]  # N
    sweep_points = 100

    # Pure Longitudinal Slip Case
    kappa_sweep = np.linspace(-0.2, 0.2, sweep_points)

    #  Pure Lateral Slip Case
    alpha_sweep = np.deg2rad( np.linspace(-15, 15, sweep_points) )  # radians

    # Meshgrid of kappa and Fz
    [KAPPA, FZ] = np.meshgrid(kappa_sweep, Fz_sweep)
    [ALPHA, _] = np.meshgrid(alpha_sweep, Fz_sweep)

    Fx_mesh = np.zeros(KAPPA.shape)
    Fy_mesh = np.zeros(ALPHA.shape)

    #  Calculate Lateral and Longitudinal Forces for each Fz
    for i in range(FZ.shape[0]):
        for j in range(KAPPA.shape[1]):
            _, Fx = simpleTyre(KAPPA[i,j], 0.0, FZ[i,j], settings)
            Fy, _ = simpleTyre(0.0, ALPHA[i,j], FZ[i,j], settings)
            Fx_mesh[i,j] = Fx
            Fy_mesh[i,j] = Fy

    # Plotting on two subplots
    plt.figure()
    plt.subplot(1,2,1)
    for i in range(Fz_sweep.__len__()):
        plt.plot(kappa_sweep, Fx_mesh[i,:], label=f'Fz={Fz_sweep[i]} N')
    plt.title('Tyre Longitudinal Force Fx vs Slip Ratio (kappa)')
    plt.xlabel('Slip Ratio (kappa)')
    plt.ylabel('Longitudinal Force Fx (N)')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    for i in range(Fz_sweep.__len__()):
        plt.plot(np.rad2deg(alpha_sweep), Fy_mesh[i,:], label=f'Fz={Fz_sweep[i]} N')
    plt.title('Tyre Lateral Force Fy vs Slip Angle (alpha)')
    plt.xlabel('Slip Angle (alpha) [degrees]')
    plt.ylabel('Lateral Force Fy (N)')
    plt.legend()
    plt.grid()
    plt.show()