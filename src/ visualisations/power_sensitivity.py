import matplotlib.pyplot as plt
import scienceplots as splt
import numpy as np

"""

Simple script to show that with increase velocity, the power sensitivity reduces

"""

mass = 280
g = 9.81
power = 80e3 # W
velocity_range = np.linspace(5, 30, 100) # m/s

power_range = np.linspace(10e3, 100e3, 5) # W

acceleration = power / (mass * velocity_range) # a = P / (m * v)


tyre_limit = 1.5 * g # m/s²

plt.style.use(['science', 'grid'])

plt.figure(figsize=(10, 6))

for power in power_range:
    acceleration = power / (mass * velocity_range) # a = P / (m * v)
    plt.plot(velocity_range, acceleration, label=f'Power = {power/1e3:.0f} kW')

plt.axhline(y=tyre_limit, color='r', linestyle='--', label=r'Tyre Limit Approximation ($\mu$ = 1.5)')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration vs Velocity')
plt.grid(True)
plt.legend()
plt.show()


