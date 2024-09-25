import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

x_grid = np.arange(-3, 3, 1)
y_grid = np.arange(-3, 3, 1)
m_range = np.arange(-10,10)
coef = 10

f_vals = m * x_grid

plt.figure(figsize=(5, 3.5))
plt.grid()
plt.plot(x_grid, m * x_grid, "b-", label=f"true line, m={m}")
plt.plot(x_grid, f_vals, "r.", label="sample data")
mpred = np.linalg.lstsq(np.array([x_grid]).T, f_vals, rcond=None)[0]
plt.plot(x_grid, mpred * x_grid, "g-", label=f"fitted line, m={round(mpred[0],3)}")
plt.legend(framealpha=1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# plt.savefig("lsqfit.eps", format="eps")
print(mpred)
