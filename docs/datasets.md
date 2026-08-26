# Datasets

Three built-in regression datasets, loaded lazily from CSV files bundled inside
the package. Each exposes `train_data` / `test_data` as
`(features: pd.DataFrame, targets: pd.Series)` tuples, plus
`predictor_names` and `response_name`.

| Dataset | Response | Predictors |
|---------|----------|------------|
| `AdvertisingDataset` | `sales` | `TV`, `radio`, `newspaper` |
| `AutoDataset` | `mpg` | all numeric columns except the response |
| `InjectionMoldingDataset` | `mass` | 8 process variables |

## Splitting

The first two use a random-free (ordered) train/test split controlled by the
`split` argument (fraction used for training, default `0.7`).
`InjectionMoldingDataset` ships a fixed 150/82 train/test split.

```python
from ml_tools.datasets import AdvertisingDataset

dataset = AdvertisingDataset(split=0.8)
x_train, y_train = dataset.train_data
x_test, y_test = dataset.test_data
# x_train : DataFrame with columns TV, radio, newspaper
# y_train : Series named "sales"
```

## Accessing the raw frames

After `train_data`/`test_data` is accessed once, the underlying DataFrames are
available as `dataset.raw_train_data` and `dataset.raw_test_data` (including
any non-feature columns).

## Notes

- `AutoDataset` coerces missing values (`"?"`) to `NaN`, converts numeric
  columns, and drops rows with missing values in both splits.
  Non-numeric columns (e.g. `name`) are excluded from the features.
- All datasets are read-only inputs: `ml-tools` never mutates them.
