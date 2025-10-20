import matplotlib.pyplot as plt
import numpy as np

# import statsmodels.api as sm

np.random.seed(42)
n_samples = 1000
mu = 0
sigma = 0.05
dof = 1

x = np.linspace(0, 1, n_samples)
y_true = x**2 + 0.5
y_pred = x + 0.5 + np.random.normal(loc=mu, scale=sigma, size=y_true.shape)

n_samples = y_true.shape[0]
residual = y_pred - y_true
fit_range = [y_pred.min(), y_pred.max()]
diff = fit_range[1] - fit_range[0]
X = x
if X.ndim == 1:
    X = X[:, None]

leverage = (X * np.linalg.pinv(X).T)[:, 0]
rss = np.sum(residual**2)
rse2 = rss / (n_samples - dof - 1)
rse = np.sqrt(rse2)
normed_residual = residual / (rse * np.sqrt(1 - leverage))
scale_loc = np.sqrt(np.abs(normed_residual))

# lowess = sm.nonparametric.lowess(
#     residual, y_pred, delta=0.1 * diff, frac=0.1, return_sorted=True
# )


fig, axs = plt.subplots(3, 2, sharex="col", constrained_layout=True)
axs[0, 0].plot(fit_range, 2 * [0], "k", linestyle="dotted")
axs[0, 0].scatter(y_pred, residual, marker="o", color=4 * (0,), edgecolor=3 * (0.6,))
# axs[0, 0].plot(lowess[:, 0], lowess[:, 1], "r")
axs[1, 0].scatter(y_pred, scale_loc, marker="o", color=4 * (0,), edgecolor=3 * (0.6,))

axs[0, 0].set_xlim(fit_range)

plt.show()
