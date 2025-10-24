"""Example of using ModelAdequacyChecker with the Auto dataset and a linear regression model with quadratic terms."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker, datasets

dataset = datasets.AutoDataset()
response_column = "mpg"
columns = [
    "horsepower",
    "weight",
]
model = LinearRegression()

raw_x, raw_y = dataset.train_data
x = raw_x[columns].to_numpy()
y_true = np.log(raw_y.to_numpy())

# Extend x with quadratic and interaction terms
x = np.hstack([x, x**2])
sq_cols = columns.copy()
for column in sq_cols:
    columns.append(f"{column}_squared")

model.fit(x, y_true)
y_pred = model.predict(x)

mac_checker = ModelAdequacyChecker()
mac_checker.analyze_sklearn(x, y_true, model, y_pred=y_pred, plot=True, predictor_names=columns)
plt.show()
