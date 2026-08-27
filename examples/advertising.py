"""Example of using ModelAdequacyChecker on the Advertising dataset with interaction terms."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import HTMLReport, ModelAdequacyChecker, datasets

parser = ArgumentParser()
parser.add_argument("-p", "--plot", action="store_true")
args = parser.parse_args()

path = Path(__file__)
report_file = path.parent / f"output/{path.stem}.html"

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
result = mac_checker.analyze_sklearn(
    x, y_true, model, y_pred=y_pred, plot=args.plot, predictor_names=columns
)
HTMLReport.from_analysis(*result).save(report_file)
plt.show()
