# Machine Learning Tools

Tools for machine learning: **model adequacy checking** (regression diagnostics) and
ready-to-use **utility datasets**.

## Features

- `ModelAdequacyChecker` (`MAC`): fits no model itself, but diagnoses an existing
  `LinearRegression` fit — standardized residuals, leverage, Cook's distance, VIF,
  residual autocorrelation — and highlights outliers, high-leverage, and influential
  points on six diagnostic plots:
  - Tukey-Anscombe, scale-location, normal Q-Q
  - sensitivity plot (leverage vs. standardized residuals with Cook's-distance contours)
  - residuals-vs-index (autocorrelation)
  - VIF bar chart
- Bundled regression datasets loaded lazily: `AdvertisingDataset`, `AutoDataset`,
  `InjectionMoldingDataset`.

## Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # dependencies + editable install
```

## Quickstart

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ml_tools import ModelAdequacyChecker
from ml_tools.datasets import AdvertisingDataset

dataset = AdvertisingDataset(split=0.8)
x, y = dataset.train_data
model = LinearRegression().fit(x, y)

checker = ModelAdequacyChecker()
metric, masks = checker.analyze_sklearn(
    x.to_numpy(),
    y.to_numpy(),
    model,
    predictor_names=list(x.columns),
)
print(metric.rse)                 # residual standard error
print(metric.pretty_vif())       # Variance Inflation Factors
plt.show()                        # six diagnostic figures
```

`analyze(...)` returns `(MetricSummary, ProblematicSampleMasks)`; pass `plot=False`
to only get the numbers. Thresholds and styling are configurable via `MACConfig`
(e.g. `MACConfig(t_threshold=3.0, cook_distance_threshold=0.5)`).

## Datasets

| Dataset                   | Response        | Split        |
| ------------------------- | --------------- | ------------ |
| `AdvertisingDataset`      | `sales`         | `split=`     |
| `AutoDataset`             | `mpg` (default) | `split=`     |
| `InjectionMoldingDataset` | `mass`          | fixed 150/82 |

```python
from ml_tools.datasets import AutoDataset

dataset = AutoDataset(response_name="mpg", split=0.7)
x_train, y_train = dataset.train_data
x_test, y_test = dataset.test_data
```

## Development

```bash
uv run ruff check .                                    # lint
uv run ty check src                                    # type check
uv run pytest --cov=ml_tools --cov-report=term-missing # tests
uv run mkdocs serve                                    # docs (auto-reload, :8000)
```

## License

See `LICENSE`.
