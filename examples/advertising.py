import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker, datasets

dataset = datasets.AdvertisingDataset()
columns = [
    "TV",
    "radio",
    # "newspaper",
]
model = LinearRegression()

raw_x, raw_y = dataset.train_data
x = raw_x[columns].to_numpy()
y_true = raw_y.to_numpy()

# Extend x with interaction term
x = np.hstack([x, np.prod(x, axis=1, keepdims=True)])
columns.append("TV_radio_interaction")

model.fit(x, y_true)
y_pred = model.predict(x)

mac_checker = ModelAdequacyChecker()
mac_checker.analyze_sklearn(x, y_true, model, y_pred=y_pred, plot=True, predictor_names=columns)
plt.show()
