"""Example of using ModelAdequacyChecker to analyze a linear regression model."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker

np.random.seed(42)
n_samples = 100
std = 0.1
special_points = [-2.5, 5, 1]
n_special = len(special_points)

x = np.random.rand(n_samples, 3)
scales = 1 + 1.7 * np.random.rand(n_samples, 2)
x[-n_special:, 0] = special_points
x[:, 2] = np.sum(scales * x[:, :2], axis=1)

y_true = x.sum(axis=1)
y_true += np.random.normal(0, std, size=y_true.shape)
y_true[-2] += 0.3
y_true[-1] += -0.45

if x.ndim == 1:
    x = x[:, None]

model = LinearRegression()
model.fit(x, y_true)
y_pred = model.predict(x)


checker = ModelAdequacyChecker()
checker.analyze_sklearn(x, y_true, model, y_pred=y_pred, plot=True)
plt.show()
