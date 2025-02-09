# G-Research Crypto Forcasting

## Solution

### Features

- basic features
  - Asset_ID
  - Count, Open, High, Low, Close, Volume, VWAP
- financial feature engineering
  - use ta-lib
- TBD
  - something other features

## Cross Validation

- Time Series Split
  - training data is 6 months
  - validation data is 3 months
  - test data is 3 month
- 3 fold split (may be more split)
- 3 random seed averaging (may be more seed)

## Model

- LightGBM + XGBoost
  - may be with lstm and transformer

### Loss Function

- Weighted Pearson Correlation

### Evaluation Function

- Weighted Pearson Correlation
