"""
Predict log nightly price with six models and compare in-/out-of-sample performance.

Input : Data/Processed/listings_clean.csv  (from Code/clean_listings.py)
Output: Results/model_performance.csv      (36 metrics: 6 models x {in, out} x {RMSE, MAE, R2})
        Results/model_performance.md       (formatted table + tuned hyperparameters)

Usage (from the project root):
    Project.venv/bin/python Code/train_models.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Processed" / "listings_clean.csv"
OUT_DIR = ROOT / "Results"

SEED = 42
TEST_SIZE = 0.2
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

TRACE_COLS = ["id", "host_id", "last_scraped", "price_quote_checkin_date", "price_quote_checkout_date",
              "first_review", "last_review"]
LABEL_COLS = ["neighbourhood_cleansed", "neighbourhood_area", "property_type", "property_group", "room_type"]
TARGET = "log_price"
# Reference levels dropped for intercept-based linear models (largest group in each block).
REFERENCE_LEVELS = ["nbhd_outer_residential_south", "prop_apartment_condo", "room_entire_home_apt"]


def metrics(y, yhat) -> dict:
    return {"RMSE": np.sqrt(mean_squared_error(y, yhat)), "MAE": mean_absolute_error(y, yhat), "R2": r2_score(y, yhat)}


def tune_nn_on_validation(X_tr, y_tr):
    """Pick MLP architecture / L2 penalty on a validation split carved from the training set."""
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)
    grid = [
        {"hidden_layer_sizes": h, "alpha": a, "learning_rate_init": lr}
        for h in [(32,), (64,), (64, 32), (128, 64)]
        for a in [1e-3, 1e-2, 1e-1, 1.0]
        for lr in [1e-3, 3e-3]
    ]
    best, best_rmse = None, np.inf
    for params in grid:
        m = make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, early_stopping=True,
                                                         n_iter_no_change=30, random_state=SEED, **params))
        m.fit(X_fit, y_fit)
        rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
        if rmse < best_rmse:
            best, best_rmse = params, rmse
    final = make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, early_stopping=True,
                                                         n_iter_no_change=30, random_state=SEED, **best))
    final.fit(X_tr, y_tr)
    return final, {**best, "validation_RMSE": round(best_rmse, 4)}


def run() -> None:
    df = pd.read_csv(DATA)
    df = df.loc[:, df.notna().any()]  # drop fully empty columns (e.g. left by editing the CSV in Excel)
    features = [c for c in df.columns if c not in TRACE_COLS + LABEL_COLS + ["price", TARGET]]
    assert len(features) == 48, f"expected 48 features, got {len(features)}"
    linear_features = [c for c in features if c not in REFERENCE_LEVELS]

    X, y = df[features], df[TARGET]
    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X, y, df["price"], test_size=TEST_SIZE, random_state=SEED)
    print(f"train={len(X_tr)}  test={len(X_te)}  features={len(features)} (linear models: {len(linear_features)})")

    models = {
        "Linear regression": (LinearRegression(), None, linear_features),
        "Ridge regression (CV)": (
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": np.logspace(-3, 3, 25)},
            linear_features,
        ),
        "CART (CV)": (
            DecisionTreeRegressor(random_state=SEED),
            {"max_depth": [3, 4, 5, 6, 8, 10, 12, None], "min_samples_leaf": [1, 5, 10, 20, 40],
             "ccp_alpha": [0.0, 1e-4, 1e-3, 5e-3]},
            features,
        ),
        "Random forest (CV)": (
            RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
            {"max_features": [0.2, 0.33, 0.5, 1.0], "min_samples_leaf": [1, 3, 5, 10]},
            features,
        ),
        "Gradient boosting (CV)": (
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [300, 600, 1000], "learning_rate": [0.01, 0.03, 0.1], "max_depth": [2, 3, 4, 5],
             "subsample": [0.8, 1.0]},
            features,
        ),
        "Neural network (V)": (None, "validation", features),
    }

    rows, tuned, dollar_rows = [], {}, []
    for name, (est, grid, cols) in models.items():
        t0 = time.time()
        if grid == "validation":
            fitted, tuned[name] = tune_nn_on_validation(X_tr[cols], y_tr)
        elif grid is None:
            fitted = est.fit(X_tr[cols], y_tr)
            tuned[name] = {}
        else:
            search = GridSearchCV(est, grid, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
            search.fit(X_tr[cols], y_tr)
            fitted = search.best_estimator_
            tuned[name] = {**search.best_params_, "CV_RMSE": round(-search.best_score_, 4)}
        pred_tr, pred_te = fitted.predict(X_tr[cols]), fitted.predict(X_te[cols])
        ins, oos = metrics(y_tr, pred_tr), metrics(y_te, pred_te)
        rows.append({"Model": name, **{f"In-sample {k}": v for k, v in ins.items()},
                     **{f"Out-of-sample {k}": v for k, v in oos.items()}})
        # supplementary: back-transformed to dollars (exp of predicted log price)
        dollar_rows.append({"Model": name, **{f"Out-of-sample {k} ($)": v
                                               for k, v in metrics(p_te, np.exp(pred_te)).items()}})
        print(f"{name:<24} done in {time.time() - t0:5.1f}s  OOS R2={oos['R2']:.3f}")

    res = pd.DataFrame(rows)
    dollars = pd.DataFrame(dollar_rows)
    OUT_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(OUT_DIR / "model_performance.csv", index=False)
    write_markdown(res, dollars, tuned, len(X_tr), len(X_te))
    print("\n" + res.round(4).to_string(index=False))


def write_markdown(res, dollars, tuned, n_tr, n_te) -> None:
    best = {c: (res[c].idxmax() if c.endswith("R2") else res[c].idxmin()) for c in res.columns if c != "Model"}
    hdr = ("| Model | RMSE | MAE | R² | RMSE | MAE | R² |\n"
           "|---|---:|---:|---:|---:|---:|---:|")
    lines = []
    for i, r in res.iterrows():
        cells = []
        for c in res.columns[1:]:
            v = f"{r[c]:.4f}" if not c.endswith("R2") else f"{r[c]:.3f}"
            cells.append(f"**{v}**" if c.startswith("Out") and best[c] == i else v)
        lines.append(f"| {r['Model']} | " + " | ".join(cells) + " |")
    L = [
        "# Model performance: log nightly price",
        "",
        f"Train/test split 80/20 (seed {SEED}): {n_tr:,} training / {n_te:,} test listings; 48 features. "
        "Metrics are on log(price). In-sample = training set; out-of-sample = held-out test set. "
        "Bold = best out-of-sample value in each column.",
        "",
        "| | In-sample performance | | | Out-of-sample performance | | |",
        hdr.split("\n")[1],
        hdr.split("\n")[0],
        *lines,
        "",
        "## Supplementary: out-of-sample error in dollars",
        "",
        "Predictions back-transformed with exp(predicted log price) and compared with actual price.",
        "",
        "| Model | RMSE ($) | MAE ($) | R² |",
        "|---|---:|---:|---:|",
        *[f"| {r['Model']} | {r.iloc[1]:.2f} | {r.iloc[2]:.2f} | {r.iloc[3]:.3f} |" for _, r in dollars.iterrows()],
        "",
        "## Tuned hyperparameters",
        "",
        *[f"- **{k}**: " + (", ".join(f"{p}={v}" for p, v in d.items()) or "none (OLS)") for k, d in tuned.items()],
        "",
        "## Notes",
        "",
        f"- Linear and ridge regression drop one reference level per one-hot block ({', '.join(REFERENCE_LEVELS)}), "
        "leaving 45 columns; tree models and the neural network use all 48.",
        "- CV models: 5-fold grid search on the training set only (scoring = RMSE). Neural network: architecture "
        "chosen on a 20% validation split of the training set, then refit on the full training set.",
        "- Median imputation in the cleaning step used the full sample (≈120 imputed values in total); "
        "the leakage this introduces into the test set is negligible but not zero.",
        "",
    ]
    (OUT_DIR / "model_performance.md").write_text("\n".join(L))


if __name__ == "__main__":
    run()
