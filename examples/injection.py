import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker, datasets

dataset = datasets.InjectionMoldingDataset()
columns = [
    "Inj1PosVolAct_Var",
    "Inj1PrsAct_meanOfInjPhase",
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
mac_checker.analyze_sklearn(x, y_true, model, y_pred=y_pred, plot=True, predictor_names=columns)
plt.show()
