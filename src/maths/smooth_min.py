import casadi as ca

def smooth_min(a, b, epsilon=1e-6):
    return (a + b) / 2 - ca.sqrt((a - b)**2 + epsilon**2) / 2