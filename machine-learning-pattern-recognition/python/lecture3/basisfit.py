import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Set up and plot the dataset
yy = np.array([1.1, 2.3, 2.9]) # N,
X = np.array([[0.8], [1.9], [3.1]]) # N,1
# plt.clf()
plt.figure(figsize=(5,3.5))
plt.grid()
# phi-functions to create various matrices of new features
# from an original matrix of 1D inputs.
def phi_linear(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin])
def phi_quadratic(Xin):
    return np.hstack([np.ones((Xin.shape[0],1)), Xin, Xin**2])
def fw_rbf(xx, cc):
    """fixed-width RBF in 1d"""
    return np.exp(-(xx-cc)**2 / 2.0)
def phi_rbf(Xin):
    return np.hstack([fw_rbf(Xin, 1), fw_rbf(Xin, 2), fw_rbf(Xin, 3)])

def fit_and_plot(phi_fn, X, yy):
    # phi_fn takes N, inputs and returns N,K basis function values
    w_fit = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0] # K,
    X_grid = np.arange(0, 4, 0.01)[:,None] # N,1
    f_grid = np.dot(phi_fn(X_grid), w_fit)
    plt.plot(X_grid, f_grid)

fit_and_plot(phi_linear, X, yy)
fit_and_plot(phi_quadratic, X, yy)
fit_and_plot(phi_rbf, X, yy)
plt.plot(X, yy, 'o')
plt.legend(('linear fit', 'quadratic fit', 'rbf fit', 'data'), framealpha=1)
plt.xlabel('$x$')
plt.ylabel('$f$')
plt.savefig("basisfit.eps",format="eps")
plt.show()
