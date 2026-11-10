# Boston Airbnb Price Lab

Predictive-analytics project that recommends a nightly price for a Boston Airbnb listing.

| Path | What it is |
|---|---|
| `data/` | Inside Airbnb Boston Sep 2026 snapshot: `listings.csv`, `neighbourhoods.csv`, `neighbourhoods.geojson` |
| `data/quarterly/` | Four quarterly snapshots (Dec 2025, Mar 2026, Jun 2026, Sep 2026). Only Jun and Sep include prices |
| `notebooks/boston_airbnb_pricing.ipynb` | Data selection, cleaning, EDA, feature engineering, five models (baseline, linear regression, Lasso, random forest, gradient boosting), comparison, verdict, export |
| `notebooks/boston_airbnb_phase3_enriched.ipynb` | Phase 3 Steps 1-2: detailed attributes (size, bathrooms, property type, amenities, review scores) and location context, with Lasso, random forest, LightGBM and CatBoost |
| `notebooks/boston_airbnb_phase3_text_market.ipynb` | Phase 3 Steps 2b-5: Zillow home values, review and description text (TF-IDF, aspect lexicon), pretrained text/photo embeddings when available |
| `notebooks/boston_airbnb_quarterly.ipynb` | Phase 2: re-select the model on June, apply it to the year, pool quarters with time + zone features, compare per-quarter weights |
| `interface/index.html` | Portfolio page with a working price calculator (open it in a browser; no server needed) |
| `interface/pricing.js` | JavaScript port of the feature builder and the five models |
| `interface/data.js` | Metrics, chart data and map exported by the notebooks (small, loads first) |
| `interface/models.js` | Trained models exported by the notebooks (loaded in the background) |
| `scripts/build_notebook.py`, `scripts/build_quarterly_notebook.py` | Sources of the notebook cells |
| `data/market/` | Zillow neighbourhood home values (ZHVI) and ZIP rents (ZORI), Boston rows |
| `data/raw/` | Not committed. Reviews, calendars and photos from `scripts/download_data.py` and `scripts/download_photos.py` |
| `data/detailed/` | Detailed Inside Airbnb exports (Mar, Jun, Sep 2026; 90 columns) |
| `src/enriched_features.py` | Phase 3 features: detailed attributes, amenity groups, landmarks, supply, local market price, optional Zillow loader |
| `src/pricing_features.py` | Shared data pre-check, cleaning rules and feature builder used by both notebooks |
| `scripts/verify_js.js`, `scripts/feature_parity.py` | Check that the browser builds the same features and predictions as Python |

## Reproduce

```bash
pip install scikit-learn pandas numpy matplotlib nbformat nbconvert ipykernel lightgbm catboost
python scripts/build_notebook.py
python scripts/build_quarterly_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_pricing.ipynb && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_quarterly.ipynb && cd ..
python scripts/build_enriched_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace boston_airbnb_phase3_enriched.ipynb && cd ..
python scripts/feature_parity.py
node scripts/verify_js.js
```

## Data pre-check and variable screening

Every row goes through `pricing_features.precheck()` (duplicates, impossible or inconsistent values, locations vs neighbourhood boundaries, missing values, rare categories). Every candidate variable is then ranked by Spearman correlation and mutual information with log price, checked for redundancy (VIF) and for leakage.

Key finding: the host's listing count acted as a host fingerprint, and a random train/test split let the model see a test listing's sibling units. All evaluation is therefore **split by host**: no host appears in both training and test data.

| Pipeline (host-grouped 5-fold CV, RMSE on log price) | Random forest | Lasso |
|---|---|---|
| Original (price filter only, all candidates) | 0.440 | 0.451 |
| + pre-check cleaning | 0.443 | 0.441 |
| + variable screening (final, 43 features) | **0.433** | 0.443 |

## Phase 1 result (Sep 2026 snapshot, 616 test listings from unseen hosts)

| Model | CV RMSE (log) | Test MAE ($/night) | R² (log) |
|---|---|---|---|
| Baseline (neighbourhood × room-type median) | 0.635 | 146 | 0.22 |
| Linear regression | 0.452 | 111 | 0.58 |
| Lasso | 0.450 | 111 | 0.58 |
| **Random forest** (chosen: tied best, most stable) | 0.449 | **104** | 0.63 |
| Gradient boosting | 0.447 | 105 | 0.63 |

## Phase 2 result (random forest, hosts held out, scored on June and September)

| Experiment | Jun MAE | Sep MAE | Sep median bias |
|---|---|---|---|
| Trained on June only | 120 | 126 | +11.9% |
| Trained on September only | 124 | 114 | +3.7% |
| Pooled Jun + Sep | 118 | 117 | +7.6% |
| **Pooled + time + zone, Sep weighted 2×** (lowest average) | 117 | 114 | +4.8% |

## Phase 3, Steps 1-2 result (Jun + Sep detailed files, 1,329 test listings from unseen hosts)

| Step | Lasso | Random forest | LightGBM | CatBoost | Best R² (log) |
|---|---|---|---|---|---|
| Phase 2 features | $141 | $135 | $136 | $135 | 0.588 |
| + Step 1 detailed attributes | $115 | $110 | **$99** | $101 | 0.760 |
| + Step 2 location context | $116 | $109 | **$98** | $103 | 0.762 |

Host-grouped 5-fold CV (LightGBM): Step 1 lowers error by $25.47/night (5 of 5 folds); landmarks and supply by $0.31 (4 of 5); the neighbours' market price makes it $0.48 worse (2 of 5). Housing-market data (Zillow) is not included: this environment's network blocks it. Put `Neighborhood_zhvi.csv` in `data/market/` and rerun.

## Phase 3, Steps 2b-5 result (LightGBM, host-grouped 5-fold CV)

| Rung | CV MAE ($/night) | Change | Folds improved |
|---|---|---|---|
| Steps 1-2 base (133 features) | 86.81 | | |
| + Zillow home values | 87.02 | +0.21 | 3 of 5 |
| + description text (TF-IDF) | 86.21 | -0.81 | 4 of 5 |
| + review aspects | 86.15 | -0.06 | 2 of 5 |
| + review text (TF-IDF) | 85.25 | -0.90 | 3 of 5 |

- Seasonality (Step 3) is not possible: Inside Airbnb calendar files no longer contain prices.
- Pretrained embeddings (bge-small for text, CLIP for photos) are coded and photos are downloaded, but model weights come from `us.aws.cdn.hf.co` (and `cas-server.xethub.hf.co` / `cas-bridge.xethub.hf.co`), which the network policy blocks.
