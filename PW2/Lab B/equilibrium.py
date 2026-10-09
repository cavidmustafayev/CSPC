import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

K = 50.0  

def k_imbalance(x):
    return ((2*x)**2)/((1-x)**2)-K

def k_imbalance_sq(x):
    x_val=x[0] if isinstance(x, (list, np.ndarray)) else x
    return k_imbalance(x_val)**2


res_root=root_scalar(k_imbalance, bracket=[0.001, 0.999], method="brentq")
x_newton=res_root.root

res_slsqp=minimize(k_imbalance_sq, x0=[0.5], method="SLSQP", bounds=[(0.001, 0.999)])
x_slsqp=res_slsqp.x[0]

print(f"Equilibrium extent x (Root-finding):{x_newton:.4f}")
print(f"Equilibrium extent x (SLSQP):{x_slsqp:.4f}")

n_H2=1.0-x_newton
n_I2=1.0-x_newton
n_HI=2.0*x_newton

print(f"Amounts H2:{n_H2:.4f}mol, I2:{n_I2:.4f} mol, HI:{n_HI:.4f} mol")

x_vals=np.linspace(0, 0.95, 200)
plt.figure(figsize=(7, 5))
plt.plot(x_vals, 1 - x_vals, label=r"$H_2$", color="blue")
plt.plot(x_vals, 1 - x_vals, label=r"$I_2$", color="green", linestyle="--")
plt.plot(x_vals, 2 * x_vals, label=r"$HI$", color="red")

plt.axvline(x=x_newton, color="gray", linestyle=":", label=f"Equilibrium (x = {x_newton:.2f})")
plt.xlabel("Reaction Extent(x)")
plt.ylabel("Amount(mol)")
plt.title(r"Chemical Equilibrium: $H_2 + I_2 \rightleftharpoons 2HI$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("equilibrium.png")
plt.close()