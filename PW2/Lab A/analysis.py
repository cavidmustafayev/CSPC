import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

#TODO1
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]

#TODO2

v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Mean acceleration: {a.mean():.2f} m/s^2")
print(f"Standard deviation of acceleration: {a.std():.2f} m/s^2")

#TODO3

v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]


#TODO4

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))


ax1.plot(t, y, label='Original Position', color='blue')
ax1.plot(t, y_rec, '--', label='Recovered Position', color='red')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)


ax2.plot(t, v, label='Velocity', color='green')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()
ax2.grid(True)


ax3.plot(t, a, label='Acceleration', color='red')
ax3.axhline(-9.81, color='black', linestyle='--', label='g = -9.81 m/s²')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('motion.png')
print("Plot saved as motion.png")