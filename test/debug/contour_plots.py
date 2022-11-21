import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mplc


delta = 0.01
x = np.arange(0, 1 + delta, delta)
y = np.arange(0, 1 + delta, delta)
X, Y = np.meshgrid(x, y)
Z1 = (X + X + Y) / 3
Z2 = np.power(X * X * Y, 1 / 3.0)
Z3 = 3 / (2 / X + 1 / Y)
Z = [Z1, Z2, Z3]
names = ["Arithmetic Mean", "Geometric Mean", "Harmonic Mean"]

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(Z),
)  # plt.figure(), plt.axes()

# fig.set_dpi(300)
fig.set_figwidth(12)
fig.tight_layout(pad=2.5)

cmap = mplc.LinearSegmentedColormap.from_list(
    "wa", [(0, (0, 0.5, 0)), (0.1, (0, 1, 0)), (0.3, (1, 1, 0)), (1, (1, 0.5, 0))], N=1000
)

# axes[0].set

for ax, zi, title in zip(axes.flat, Z, names):
    ax.set_title(title)
    ax.set_box_aspect(1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.25, 0.25))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_xlabel("Descriptor Distance")
    ax.set_ylabel("Spatial Distance")

    CS = ax.contourf(X, Y, zi, levels=np.arange(0, 1 + delta, delta), cmap=cmap)
    CS2 = ax.contour(X, Y, zi, levels=[-1, 0.25, 1], linestyles="dashed", colors="k")
    ax.clabel(CS2, fmt="%2.2f", colors="k", fontsize=11)


cb = fig.colorbar(
    CS, pad=0.25, ticks=np.arange(0, 1.1, 0.1), ax=axes, location="bottom", aspect=20 * 0.15 / 0.06 * 0.8, fraction=0.06
)
cb.ax.plot([0.25, 0.25], [0, 1], "k")


plt.savefig("data/out/thesis/contour_all.jpg", bbox_inches="tight", dpi=300)
plt.show()
