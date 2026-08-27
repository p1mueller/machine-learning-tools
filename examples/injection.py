"""Example of using ModelAdequacyChecker with an injection molding dataset and a linear regression model."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ml_tools import HTMLReport, ModelAdequacyChecker, datasets

parser = ArgumentParser()
parser.add_argument("-p", "--plot", action="store_true")
args = parser.parse_args()

path = Path(__file__)
report_file = path.parent / f"output/{path.stem}.html"
dataset = datasets.InjectionMoldingDataset()
columns = [
    "Inj1PosVolAct_Var",
    # "Inj1PrsAct_meanOfInjPhase",
    "Inj1HtgEd3Act_1stPCscore",
    "ClpFceAct_1stPCscore",
    "OilTmp1Act_1stPCscore",
]
model = LinearRegression()

raw_x, raw_y = dataset.train_data
x = raw_x[columns].to_numpy()
y_true = raw_y.to_numpy()
model.fit(x, y_true)
y_pred = model.predict(x)


mac_checker = ModelAdequacyChecker()
result = mac_checker.analyze_sklearn(
    x, y_true, model, y_pred=y_pred, plot=args.plot, predictor_names=columns
)
HTMLReport.from_analysis(*result).save(report_file)
plt.show()
