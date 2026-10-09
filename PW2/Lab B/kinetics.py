import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

data = pd.read_csv("kinetics.csv")
t = data["time"].values
C_meas = data["concentration"].values

C0 = C_meas[0]

def total_error(k):
    C_pred = C0 * np.exp(-k * t)
    return np.sum((C_meas - C_pred) ** 2)

res = minimize(
    lambda k: total_error(k[0]), x0=[0.5], method="SLSQP", bounds=[(0, 5)]
)
k_fitted = res.x[0]

print(f"Fitted rate constant k = {k_fitted:.4f}")

t_dense = np.linspace(t.min(), t.max(), 200)
C_dense = C0 * np.exp(-k_fitted * t_dense)

plt.figure(figsize=(7, 5))
plt.scatter(t, C_meas, color="red", label="Measured Data", zorder=3)
plt.plot(t_dense, C_dense, color="blue", label=f"Fitted Curve (k = {k_fitted:.4f})")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (C)")
plt.title("First-Order Kinetics Rate Constant Fitting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kinetics.png")
plt.close()

