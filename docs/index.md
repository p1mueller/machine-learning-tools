# ml-tools

**Model adequacy checking and utility datasets for linear regression.**

`ml-tools` fits no models of its own. Give it a fitted scikit-learn
`LinearRegression`, and it returns the full battery of regression diagnostics —
standardized residuals, leverage, Cook's distance, VIF, residual autocorrelation —
plus flags for outliers, high-leverage and influential points, and a set of
publication-ready diagnostic plots.

## Features

- **One-call analysis** — `[ModelAdequacyChecker][ml_tools.ModelAdequacyChecker]`
  (`MAC`) turns a fitted model into metrics, masks and plots.
- **Diagnostic plots** — Tukey-Anscombe, scale-location, normal Q-Q, sensitivity
  (Cook's-distance contours), residual autocorrelation, and VIF bar chart.
- **Built-in datasets** — `[AdvertisingDataset][ml_tools.datasets.AdvertisingDataset]`,
  `[AutoDataset][ml_tools.datasets.AutoDataset]` and
  `[InjectionMoldingDataset][ml_tools.datasets.InjectionMoldingDataset]`,
  loaded lazily and split into train/test.

## Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url> ml-tools
cd ml-tools
uv sync          # installs dependencies + an editable copy of the package
```

## Next steps

- Walk through a full example in the [Getting started](getting_started.md) guide.
- See the bundled [Datasets](datasets.md) and pick one to analyze.
- Browse the generated [API reference](api.md).
