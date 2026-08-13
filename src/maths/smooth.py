import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def smooth_max(a, b, k=1e-3):
    """Tanh-based smooth approximation of max(a, b).

    Formula:
        d = a - b
        s = tanh(d / k)
        max ≈ 0.5*(a + b) + 0.5*d*s

    k controls the transition region: smaller k -> sharper transition.
    Choose k relative to the expected scale of (a - b).
    """
    d = a - b
    s = ca.tanh(d / k)
    return 0.5 * (a + b) + 0.5 * d * s


def smooth_min(a, b, k=1e-3):
    """Tanh-based smooth approximation of min(a, b).

    Uses the same k parameter as `smooth_max` and formula:
        min ≈ 0.5*(a + b) - 0.5*d*s
    """
    d = a - b
    s = ca.tanh(d / k)
    return 0.5 * (a + b) - 0.5 * d * s


def smooth_step(x, k=1e3):
    """Smooth Tanh approximation to the step / indicator function using a
    logistic (sigmoid) shaped transition.

    Suggested values for k based on the magnitude of x:
    - For x in the range of 10000 to 100000, use k = 0.1
    - For x in the range of 100000 to 1000000, use k = 0.01
    """
    # logistic function: 1 / (1 + exp(-k*x))
    return 0.5 * (1 + ca.tanh(x / k))


if __name__ == "__main__":

    # # Test Harness for smooth_step function
    # x = np.linspace(-350e3, 350e3, 10000)
    # y = [smooth_step(xi - 100e3, k=0.00001) for xi in x]

    # plt.plot(x, y)
    # plt.title("Smooth Step Function")
    # plt.xlabel("Input")
    # plt.ylabel("Output")
    # plt.grid()
    # plt.show()

    # Test Harness for smooth_max and smooth_min functions
    a = np.linspace(-10, 10, 100)
    max_vals = [smooth_max(ai, 0, k=1) for ai in a]
    min_vals = [smooth_min(ai, 0, k=1) for ai in a]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(a, max_vals, label='Smooth Max', color='blue')
    plt.plot(a, a, label='a', color='orange', linestyle='--')
    plt.title("Smooth Max Function")
    plt.xlabel("a")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(a, min_vals, label='Smooth Min', color='green')
    plt.plot(a, a, label='a', color='orange', linestyle='--')
    plt.title("Smooth Min Function")
    plt.xlabel("a")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.show()