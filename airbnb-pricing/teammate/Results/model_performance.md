# Model performance: log nightly price

Train/test split 80/20 (seed 42): 2,401 training / 601 test listings; 48 features. Metrics are on log(price). In-sample = training set; out-of-sample = held-out test set. Bold = best out-of-sample value in each column.

| | In-sample performance | | | Out-of-sample performance | | |
|---|---:|---:|---:|---:|---:|---:|
| Model | RMSE | MAE | R² | RMSE | MAE | R² |
| Linear regression | 0.3783 | 0.2965 | 0.735 | 0.3838 | 0.2974 | 0.727 |
| Ridge regression (CV) | 0.3785 | 0.2967 | 0.735 | 0.3834 | 0.2970 | 0.728 |
| CART (CV) | 0.3031 | 0.2333 | 0.830 | 0.4046 | 0.3048 | 0.697 |
| Random forest (CV) | 0.1143 | 0.0856 | 0.976 | 0.3109 | 0.2267 | 0.821 |
| Gradient boosting (CV) | 0.1272 | 0.0999 | 0.970 | **0.2988** | **0.2207** | **0.835** |
| Neural network (V) | 0.2179 | 0.1619 | 0.912 | 0.3383 | 0.2507 | 0.788 |

## Supplementary: out-of-sample error in dollars

Predictions back-transformed with exp(predicted log price) and compared with actual price.

| Model | RMSE ($) | MAE ($) | R² |
|---|---:|---:|---:|
| Linear regression | 189.47 | 89.42 | 0.535 |
| Ridge regression (CV) | 191.91 | 89.47 | 0.523 |
| CART (CV) | 212.99 | 98.89 | 0.412 |
| Random forest (CV) | 193.05 | 75.64 | 0.517 |
| Gradient boosting (CV) | 186.13 | 73.63 | 0.551 |
| Neural network (V) | 260.96 | 88.66 | 0.118 |

## Tuned hyperparameters

- **Linear regression**: none (OLS)
- **Ridge regression (CV)**: ridge__alpha=31.622776601683793, CV_RMSE=0.3904
- **CART (CV)**: ccp_alpha=0.0001, max_depth=8, min_samples_leaf=10, CV_RMSE=0.403
- **Random forest (CV)**: max_features=0.33, min_samples_leaf=1, CV_RMSE=0.3136
- **Gradient boosting (CV)**: learning_rate=0.03, max_depth=5, n_estimators=600, subsample=0.8, CV_RMSE=0.3004
- **Neural network (V)**: hidden_layer_sizes=(128, 64), alpha=1.0, learning_rate_init=0.001, validation_RMSE=0.3211

## Notes

- Linear and ridge regression drop one reference level per one-hot block (nbhd_outer_residential_south, prop_apartment_condo, room_entire_home_apt), leaving 45 columns; tree models and the neural network use all 48.
- CV models: 5-fold grid search on the training set only (scoring = RMSE). Neural network: architecture chosen on a 20% validation split of the training set, then refit on the full training set.
- Median imputation in the cleaning step used the full sample (≈120 imputed values in total); the leakage this introduces into the test set is negligible but not zero.
