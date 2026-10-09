import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("titration.csv")
V=data["volume_base"].values
pH=data["pH"].values

dpH_dV=np.gradient(pH, V)

eq_index=np.argmax(dpH_dV)
v_eq=V[eq_index]

print(f"Titration Equivalence Point:V_base={v_eq:.2f}mL (pH = {pH[eq_index]:.2f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(V, pH, color="purple", linewidth=2, label="pH Curve")
ax1.axvline(x=v_eq, color="red", linestyle="--", label=f"Eq. Point ({v_eq:.1f} mL)")
ax1.set_xlabel("Volume of Base (mL)")
ax1.set_ylabel("pH")
ax1.set_title("Titration Curve")
ax1.grid(True)
ax1.legend()

ax2.plot(V, dpH_dV, color="blue", linewidth=2, label=r"$\frac{dpH}{dV}$")
ax2.axvline(x=v_eq, color="red", linestyle="--", label=f"Peak Slope ({v_eq:.1f} mL)")
ax2.set_xlabel("Volume of Base (mL)")
ax2.set_ylabel("Slope (dpH / dV)")
ax2.set_title("First Derivative (Steepness)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("titration.png")
plt.close()