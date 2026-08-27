"""Example of using ModelAdequacyChecker to analyze a linear regression model.

Produces one report in each supported output format:

- ``show_case.html``  — self-contained HTML document with the diagnostic plots
                     embedded as inline PNGs
- ``show_case.md``    — Markdown report (for docs / README)
- ``show_case.txt``   — plain-text report (for console output)
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from ml_tools import HTMLReport, MarkdownReport, ModelAdequacyChecker, TextReport

parser = ArgumentParser()
parser.add_argument("-p", "--plot", action="store_true")
args = parser.parse_args()

path = Path(__file__)
report_file = path.parent / f"output/{path.stem}.html"
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

# 1. Run the adequacy analysis once; the plotters stay unrendered here.
metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(
    x, y_true, model, y_pred=y_pred, plot=args.plot
)

# 2. Build reports. from_analysis renders the diagnostic plots into
#    base64 PNGs for embedding — so only the HTML report pays that cost.
html = HTMLReport.from_analysis(metric, masks, plots)
html.save(report_file)

# 3. Convert between adapters without re-rendering the figures.
md = MarkdownReport.model_validate(html.model_dump())
md.save(report_file.with_suffix(".md"))

text = TextReport.model_validate(html.model_dump())
text.save(report_file.with_suffix(".txt"))
print(text.render())
plt.show()
