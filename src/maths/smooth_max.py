import casadi as ca

def smooth_max(a, b, epsilon=1e-3):
    return (a + b) / 2 + ca.sqrt((a - b)**2 + epsilon**2) / 2