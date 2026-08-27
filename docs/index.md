# ml-tools

**Model adequacy checking and utility datasets for linear regression.**

`ml-tools` fits no models of its own. Give it a fitted scikit-learn
`LinearRegression`, and it returns the full battery of regression diagnostics —
standardized residuals, leverage, Cook's distance, VIF, residual autocorrelation —
plus flags for outliers, high-leverage and influential points, and a set of
publication-ready diagnostic plots.

## Features

- **One-call analysis** —
  [ModelAdequacyChecker](reference/ml_tools/mac/analyze/#ml_tools.mac.analyze.ModelAdequacyChecker) (`MAC`) turns a fitted
  model into metrics, masks and plots.
- **Diagnostic plots** — Tukey-Anscombe, scale-location, normal Q-Q, sensitivity
  (Cook's-distance contours), residual autocorrelation, and VIF bar chart.
- **Reports** —
  [TextReport](reference/ml_tools/mac/report/),
  [MarkdownReport](reference/ml_tools/mac/report/) and
  [HTMLReport](reference/ml_tools/mac/report/) turn any analysis into a
  shareable report, with the diagnostic plots embedded as inline images in
  HTML.
- **Built-in datasets** —
  [AdvertisingDataset](reference/ml_tools/datasets/advertising/),
  [AutoDataset](reference/ml_tools/datasets/auto/) and
  [InjectionMoldingDataset](reference/ml_tools/datasets/injection_molding/),
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
