import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Note: Do NOT rename variables 'real' or 'imag' -> set from C code
plt.figure()
plt.title(r"Roots of polynomial")
plt.xlabel(r"$\mathsf{Re}$")
plt.ylabel(r"$\mathsf{Im}$", rotation = 0)
plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
colors = cm.rainbow(np.linspace(0, 1, len(real)))
plt.grid(True)
for x, y, color in zip(real, imag, colors):
    plt.plot(x, y, 'o', color=color, markersize=8)
plt.show()
plt.close()
