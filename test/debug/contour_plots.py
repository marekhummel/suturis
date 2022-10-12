import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


delta = 0.01
x = np.arange(0, 1, delta)
y = np.arange(0, 1, delta)
X, Y = np.meshgrid(x, y)
Z1 = (X + X + Y) / 3
Z2 = np.power(X * X * Y, 1 / 3.0)
Z3 = 3 / (2 / X + 1 / Y)

fig, ax = plt.subplots(1, 3)
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[2].set_box_aspect(1)
CS1 = ax[0].contourf(X, Y, Z1, levels=np.arange(0, 1, 0.01))
CS2 = ax[1].contourf(X, Y, Z2, levels=np.arange(0, 1, 0.01))
CS3 = ax[2].contourf(X, Y, Z3, levels=np.arange(0, 1, 0.01))
# ax.clabel(CS, inline=True, fontsize=10)
plt.show()
