import numpy as np
from math import atan, sin, pi

def tyreParameters():
    params = dict()
    params['tyre'] = dict()
    params['tyre']['reference_load_1'] = 2000
    params['tyre']['reference_load_2'] = 6000
    params['tyre']['peak_mux_reference_load_1'] = 1.75
    params['tyre']['peak_mux_reference_load_2'] = 1.40
    params['tyre']['peak_mux_slip_load_1'] = 0.11
    params['tyre']['peak_mux_slip_load_2'] = 0.10
    params['tyre']['peak_muy_reference_load_1'] = 1.80
    params['tyre']['peak_muy_reference_load_2'] = 1.45
    params['tyre']['peak_muy_slip_load_1'] = 9 # degrees
    params['tyre']['peak_muy_slip_load_2'] = 8 # degrees
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

    kappa_norm = kappa/kappa_max
    alpha_norm = alpha/alpha_max

    # Combined Slip Coefficient

    rho = np.sqrt(alpha_norm**2 + kappa_norm**2 + 1e-3)

    Sx = pi/(2*atan(Qx))
    Sy = pi/(2*atan(Qy))

    mux = mux_max * sin(Qx * atan(Sx * rho))
    muy = muy_max * sin(Qy * atan(Sy * rho))

    Fx = mux * Fz * kappa_norm/rho
    Fy = -muy * Fz * alpha_norm/rho

    return Fy, Fx

if __name__ == "__main__":

    settings = tyreParameters()

    Fz = 4000  # N
    kappa = 0.05  # Longitudinal Slip
    alpha = 5 * (pi/180)  # Slip Angle in Radians

    Fy, Fx = simpleTyre(kappa, alpha, Fz, settings)

    print("Tyre Forces:")
    print("Longitudinal Force Fx: ", Fx)
    print("Lateral Force Fy: ", Fy)