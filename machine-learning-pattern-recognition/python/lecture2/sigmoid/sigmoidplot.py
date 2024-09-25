import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def sigmoid(xx, vv, bb):
    return 1/(1+np.exp(-vv*xx - bb))


x_grid = np.arange(-10, 10, 0.01)
bvals = [-5, 0, 3]
vvals = [2, 1, 0.7]
plt.figure(figsize=(5, 3.5))
plt.grid()
for v, b in zip(vvals, bvals):
    plt.plot(x_grid, sigmoid(x_grid, v, b), "-", label=f"$v={v}$, $b={b}$")
plt.legend(framealpha=1)
plt.xlabel("$x$")
plt.ylabel("$y$")
# plt.show()
plt.savefig("sigmoidplot.eps", format="eps")
