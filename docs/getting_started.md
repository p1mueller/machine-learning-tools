# Getting started

A complete, runnable walkthrough. This mirrors `examples/injection.py`.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker
from ml_tools.datasets import InjectionMoldingDataset

# 1. Load a dataset (train/test split is done for you)
dataset = InjectionMoldingDataset()
x, y = dataset.train_data

# 2. Fit an ordinary least-squares model
model = LinearRegression().fit(x, y)

# 3. Analyze: metrics + problem-point masks, and six diagnostic plots
metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(
    x, y, model,
    predictor_names=list(x.columns),
)

# 4. Inspect the results
print(f"Residual standard error: {metric.rse:.3f}")
print(metric.pretty_vif())          # Variance Inflation Factors
print(f"Flagged points: {int(masks.combined.sum())} of {len(masks.combined)}")
print(f"  - outliers       : {int(masks.outliers.sum())}")
print(f"  - high leverage  : {int(masks.high_leverage.sum())}")
print(f"  - influential    : {int(masks.influential.sum())}")

plt.show()                          # six diagnostic figures
```

## What you get back

| Object | Type | Contents |
|--------|------|----------|
| `metric` | `MetricSummary` | `r_squared`, `adj_r_squared`, `f_statistic`, `rse`, `rss`, `leverage`, `standardized_residuals`, `cooks_distance`, `vif`, `residual_correlation`, `pretty_vif()` |
| `masks` | `ProblematicSampleMasks` | boolean arrays `outliers`, `high_leverage`, `influential` and their union `combined` |
| `plots` | `DiagnosticPlots` | six unrendered plotters: `residuals`, `scale_location`, `qq`, `sensitivity`, `residual_correlation`, `vif` |

## Reports

The analysis results can be turned into a shareable report in plain text,
Markdown or HTML:

```python
from ml_tools import HTMLReport, MarkdownReport, TextReport

# from_analysis renders the diagnostic plots into base64 PNGs,
# so this is the only step that costs figure rendering
html = HTMLReport.from_analysis(metric, masks, plots)
html.save("report.html")       # self-contained document (inline figures)

md = MarkdownReport.model_validate(html.model_dump())   # no re-rendering
md.save("report.md")

text = TextReport.model_validate(html.model_dump())
text.save("report.txt")
print(text.render())
```

What the report contains:

| Section | Contents |
|---------|----------|
| **Model statistics** | `n_samples`, `n_features`, dof, $R^2$, adjusted $R^2$, RSE, RSS, TSS, F-statistic, residual autocorrelation |
| **Predictors** | per-predictor VIF, flagged where `> vif_threshold` |
| **Problematic samples** | indices of outliers, high-leverage and influential points |
| **Diagnostic plots** | (HTML only) the six figures embedded as inline PNGs |

The report is a pydantic model, so you can also use `report.to_dict()`,
`report.to_dataframe()` or mutate it before rendering. Pick the adapter
subclass to match the output you want — each implements `render()` and
`save(path)`.

The six figures are: Tukey-Anscombe (residuals vs. fitted), scale-location,
normal Q-Q, sensitivity (leverage vs. standardized residuals with Cook's-distance
contours), residuals-vs-index, and the VIF bar chart.

With `plot=True` (default) the figures are rendered immediately. With
`plot=False` the same `DiagnosticPlots` object is returned but nothing is
rendered — call `plots.plot_all(masks)` later, wherever you need the figures:

```python
metric, masks, plots = ModelAdequacyChecker().analyze_sklearn(x, y, model, plot=False)
...
plots.plot_all(masks)   # render all six figures now
plt.show()
```

## Configuring thresholds and styling

All defaults live in `MACConfig`:

```python
from ml_tools import MACConfig

config = MACConfig(
    t_threshold=3.0,                # |standardized residual| beyond which = outlier
    cook_distance_threshold=0.5,    # Cook's distance beyond which = influential
    leverage_threshold_factor=4.0,  # factor over dof/n for the high-leverage cut
)
metric, masks, plots = ModelAdequacyChecker(config=config).analyze_sklearn(x, y, model)
```

`MACConfig.colors` controls plot colors, grid lines, and marker style.

## Using your own fit

`analyze_sklearn` only needs the model's coefficient vector count for
degree-of-freedom calculation, so it works with any fitted scikit-learn
`LinearRegression` — no need to use `ml-tools` datasets:

```python
import pandas as pd

df = pd.read_csv("my_data.csv")
x = df[["a", "b", "c"]].to_numpy()
y = df["target"].to_numpy()
model = LinearRegression().fit(x, y)
ModelAdequacyChecker().analyze_sklearn(x, y, model)
```
