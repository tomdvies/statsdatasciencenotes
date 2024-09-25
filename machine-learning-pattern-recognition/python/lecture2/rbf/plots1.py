import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def rbf_1d(xx, cc, hh):
    return np.exp(-((xx - cc) ** 2) / hh**2)


x_grid = np.arange(-3, 5, 0.01)
cvals = [1, 2, 3]
h2vals = [2, 1, 0.1]
plt.figure(figsize=(5, 3.5))
plt.grid()
for c, h2 in zip(cvals, h2vals):
    plt.plot(x_grid, rbf_1d(x_grid, c, h2), "-", label=f"$c={c}$, $h^2={h2}$")
plt.legend(framealpha=1)
plt.xlabel("$x$")
plt.ylabel("$y$")
# plt.show()
plt.savefig("rbfplot.eps", format="eps")
