# Boston Airbnb Price Lab

Predictive-analytics project that recommends a nightly price for a Boston Airbnb listing.

| Path | What it is |
|---|---|
| `data/` | Inside Airbnb Boston snapshot: `listings.csv`, `neighbourhoods.csv`, `neighbourhoods.geojson` |
| `notebooks/boston_airbnb_pricing.ipynb` | Data selection, cleaning, EDA, feature engineering, five models (baseline, linear regression, Lasso, random forest, gradient boosting), comparison, verdict, export |
| `interface/index.html` | Portfolio page with a working price calculator (open it in a browser; no server needed) |
| `interface/pricing.js` | JavaScript port of the feature builder and the five models |
| `interface/data.js` | Trained models + metrics exported by the notebook |
| `scripts/build_notebook.py` | Source of the notebook cells |
| `scripts/verify_js.js` | Checks that the browser predictions match Python |

## Reproduce

```bash
pip install scikit-learn pandas numpy matplotlib nbformat nbconvert ipykernel
python scripts/build_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_pricing.ipynb && cd ..
node scripts/verify_js.js
```

## Result (held-out test set, 696 listings)

| Model | Test MAE ($/night) | R² (log price) |
|---|---|---|
| Baseline (neighbourhood × room-type median) | 128 | 0.40 |
| Linear regression | 104 | 0.68 |
| Lasso | 104 | 0.68 |
| **Random forest** | **86** | **0.79** |
| Gradient boosting | 88 | 0.78 |
