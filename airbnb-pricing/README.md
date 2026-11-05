# Boston Airbnb Price Lab

Predictive-analytics project that recommends a nightly price for a Boston Airbnb listing.

| Path | What it is |
|---|---|
| `data/` | Inside Airbnb Boston Sep 2026 snapshot: `listings.csv`, `neighbourhoods.csv`, `neighbourhoods.geojson` |
| `data/quarterly/` | Four quarterly snapshots (Dec 2025, Mar 2026, Jun 2026, Sep 2026). Only Jun and Sep include prices |
| `notebooks/boston_airbnb_pricing.ipynb` | Data selection, cleaning, EDA, feature engineering, five models (baseline, linear regression, Lasso, random forest, gradient boosting), comparison, verdict, export |
| `notebooks/boston_airbnb_quarterly.ipynb` | Phase 2: re-select the model on June, apply it to the year, pool quarters with time + zone features, compare per-quarter weights |
| `interface/index.html` | Portfolio page with a working price calculator (open it in a browser; no server needed) |
| `interface/pricing.js` | JavaScript port of the feature builder and the five models |
| `interface/data.js` | Metrics, chart data and map exported by the notebooks (small, loads first) |
| `interface/models.js` | Trained models exported by the notebooks (loaded in the background) |
| `scripts/build_notebook.py`, `scripts/build_quarterly_notebook.py` | Sources of the notebook cells |
| `scripts/verify_js.js` | Checks that the browser predictions match Python |

## Reproduce

```bash
pip install scikit-learn pandas numpy matplotlib nbformat nbconvert ipykernel
python scripts/build_notebook.py
python scripts/build_quarterly_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_pricing.ipynb && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_quarterly.ipynb && cd ..
node scripts/verify_js.js
```

## Phase 1 result (Sep 2026 snapshot, held-out test set, 696 listings)

| Model | Test MAE ($/night) | R² (log price) |
|---|---|---|
| Baseline (neighbourhood × room-type median) | 128 | 0.40 |
| Linear regression | 104 | 0.68 |
| Lasso | 104 | 0.68 |
| **Random forest** | **86** | **0.79** |
| Gradient boosting | 88 | 0.78 |

## Phase 2 result (random forest, listings held out by ID, scored on September)

| Experiment | Sep MAE ($/night) | Sep median bias |
|---|---|---|
| Trained on June only | 101 | +12.5% |
| Pooled Jun + Sep | 87 | +7.1% |
| Pooled + time + zone, Sep weighted 2× | **83** | +5.1% |
