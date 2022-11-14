import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mplc


delta = 0.01
x = np.arange(0, 1 + delta, delta)
y = np.arange(0, 1 + delta, delta)
X, Y = np.meshgrid(x, y)
# Z = (X + X + Y) / 3
# Z = np.power(X * X * Y, 1 / 3.0)
Z = 3 / (2 / X + 1 / Y)

fig, ax = plt.figure(), plt.axes()
cmap = mplc.LinearSegmentedColormap.from_list(
    "wa", [(0, (0, 0.5, 0)), (0.1, (0, 1, 0)), (0.3, (1, 1, 0)), (1, (1, 0.5, 0))], N=1000
)
fig.tight_layout()
ax.set_box_aspect(1)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
ax.set_xlabel("Descriptor Distance")
ax.set_ylabel("Spatial Distance")


CS = ax.contourf(X, Y, Z, levels=np.arange(0, 1 + delta, delta), cmap=cmap)
CS2 = ax.contour(X, Y, Z, levels=[-1, 0.25, 1], linestyles="dashed", colors="k")

ax.clabel(CS2, fmt="%2.2f", colors="k", fontsize=11)

cb = fig.colorbar(CS, pad=0.1, ticks=np.arange(0, 1.1, 0.1))
cb.ax.plot([0, 1], [0.25, 0.25], "k")


plt.savefig("data/out/thesis/contour_harm.jpg", bbox_inches="tight")
# plt.show()
