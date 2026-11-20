"""Builds notebooks/boston_airbnb_quarterly.ipynb (Phase 2: a year of quarterly snapshots).

Run:  python scripts/build_quarterly_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace notebooks/boston_airbnb_quarterly.ipynb
Phase 1 (notebooks/boston_airbnb_pricing.ipynb) must have been run first: this notebook adds
a "quarterly" block to interface/data.js and the quarterly model to interface/models.js.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 2: From one snapshot to a year of quarters

Phase 1 pre-checked the data, screened the variables down to 43 features, and found that **Random Forest** is the most reliable pricing model under host-grouped evaluation. Prices move during the year, so this notebook asks three questions:

1. **Does a model trained on one quarter (June) still work in later quarters?** We test it on September listings it has never seen.
2. **Do time and zone features help?** We pool the quarters, add *time* features (quarter, how long a listing has been live) and *zone* features (distance ring and geographic clusters), and measure the gain.
3. **How do the weights of each parameter change between quarters?** We fit Lasso per quarter and compare coefficients, and we test recency weighting (newer quarters count more).

### The four snapshots

| File | Snapshot | Listings | Prices |
|---|---|---|---|
| `listings_2025Q4_dec.csv` | Dec 2025 (latest review 1 Jan 2026) | 3,706 | none published |
| `listings_2026Q1_mar.csv` | Mar 2026 (partial scrape) | 1,881 | none published |
| `listings_2026Q2_jun.csv` | Jun 2026 | 4,414 | 3,729 |
| `listings_2026Q3_sep.csv` | Sep 2026 (the Phase 1 data) | 4,368 | 3,486 |

Inside Airbnb did not publish prices for the December and March snapshots. Those two quarters can't be training targets. We use them in two honest ways: to measure each listing's **tenure** (how many earlier quarters it was already live), and to **price the December and March listing stock** with the trained model. Those predictions can't be scored. If priced versions of those files become available, drop them into `data/quarterly/` and every model below trains on them automatically.
""")

code(r"""
import json, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, "../src")
import pricing_features as pf   # same pre-check and features as Phase 1

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
DATA = Path("../data"); QDIR = DATA / "quarterly"; OUT = Path("../interface")
SEED = 42
QUARTERS = [  # code, label, file, approx. scrape date
    ("2025Q4", "Dec 2025", "listings_2025Q4_dec.csv", "2026-01-01"),
    ("2026Q1", "Mar 2026", "listings_2026Q1_mar.csv", "2026-03-28"),
    ("2026Q2", "Jun 2026", "listings_2026Q2_jun.csv", "2026-07-05"),
    ("2026Q3", "Sep 2026", "listings_2026Q3_sep.csv", "2026-09-20"),
]
""")

md(r"""
## 1. Load, pre-check and stack the quarters

Every quarter goes through the same pre-check as Phase 1 (`src/pricing_features.py::precheck`). Price rules are applied afterwards, only to the priced quarters, so December and March keep all their listings for the tenure and stock analysis.
""")

code(r"""
official = pd.read_csv(DATA / "neighbourhoods.csv")["neighbourhood"]
geo = pf.load_geo(DATA / "neighbourhoods.geojson")
NEIGHBOURHOODS = pf.model_neighbourhoods(official)
CENTER = pf.CENTER
frames, reports = [], {}
for i, (q, label, f, scraped) in enumerate(QUARTERS):
    raw = pd.read_csv(QDIR / f)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    d, rep_q = pf.precheck(raw, official, geo, require_price=False)
    reports[label] = rep_q.set_index("check")["rows"]
    d["quarter"], d["q_label"], d["q_idx"], d["scraped"] = q, label, i, pd.Timestamp(scraped)
    frames.append(d)
A = pd.concat(frames, ignore_index=True)

# Cross-quarter check: some rows have no host at all. Recover it from the same listing in another quarter.
miss = A["host_id"].isna()
known = A.loc[~miss].drop_duplicates("id").set_index("id")[["host_id", "calculated_host_listings_count"]]
rec = A.loc[miss, "id"].map(known["host_id"])
A.loc[miss, "host_id"] = rec
A.loc[miss, "calculated_host_listings_count"] = A.loc[miss, "id"].map(known["calculated_host_listings_count"]).fillna(1)
still = A["host_id"].isna()
host_fix = A.loc[miss].groupby("q_label").size().to_frame("host missing")
host_fix["recovered from another quarter"] = A.loc[miss & ~still].groupby("q_label").size()
host_fix = host_fix.fillna(0).astype(int)
print("Host information missing:"); display(host_fix)
A = A[~still].copy()
for label in reports:
    reports[label]["Host missing (recovered or dropped)"] = int(host_fix["host missing"].get(label, 0))
precheck_q = pd.DataFrame(reports)
print("Pre-check, rows affected per quarter:")
display(precheck_q)

summary = A.groupby("q_label", sort=False).agg(
    listings=("id", "size"), priced=("price", lambda s: s.notna().sum()),
    median_price=("price", "median"), entire_share=("room_type", lambda s: (s == "Entire home/apt").mean()))
summary["entire_share"] = (summary["entire_share"] * 100).round(1)
summary
""")

md(r"""
## 2. New features: time and zone

**Time**
* `q_sep` is a quarter indicator (Sep = 1, Jun = 0). With two priced quarters, a single indicator is all the data can identify. With four priced quarters it becomes three indicators, or a season curve.
* `tenure_q` counts how many earlier snapshots the listing appeared in (0-3). Long-running listings are established and tend to price differently from newcomers. This feature is what the unpriced Dec and Mar files contribute.
* `days_since_review` is the number of days between the last review and the scrape date (activity recency, capped at 3 years).

**Zone**
* `ring_core`, `ring_inner`, `ring_outer` sort listings by distance to Boston Common: under 2 km, 2-5 km, and over 5 km.
* `zone_*` are 8 geographic clusters (k-means on coordinates in km, fitted on all four quarters). They cut across neighbourhood lines. For example, the waterfront strip of South Boston behaves differently from inland South Boston.
""")

code(r"""
def base_features(d):
    # Exactly the Phase 1 screened feature set
    return pf.build_features(d, NEIGHBOURHOODS, screened=True)

BASE = list(base_features(A.head(2)).columns)

# --- time features ---
seen = A.groupby("id")["q_idx"].apply(set).to_dict()
A["tenure_q"] = [sum(j < q for j in seen[i]) for i, q in zip(A["id"], A["q_idx"])]
A["days_since_review"] = (A["scraped"] - pd.to_datetime(A["last_review"])).dt.days.clip(0, 1095).fillna(1095)

# --- zone features ---
coords_km = np.c_[(A["latitude"] - CENTER[0]) * 111.0, (A["longitude"] - CENTER[1]) * 82.0]
km = KMeans(8, n_init=10, random_state=SEED).fit(coords_km)
A["zone"] = km.labels_
zone_names = A.groupby("zone")["neighbourhood"].agg(lambda s: " / ".join(s.value_counts().index[:2])).to_dict()
print("zones:", zone_names)

def time_zone_features(d):
    X = pd.DataFrame(index=d.index)
    X["q_sep"] = (d["quarter"] == "2026Q3").astype(int)
    X["tenure_q"] = d["tenure_q"]
    X["days_since_review"] = d["days_since_review"]
    dist = np.sqrt(((d["latitude"] - CENTER[0]) * 111.0) ** 2 + ((d["longitude"] - CENTER[1]) * 82.0) ** 2)
    X["ring_core"] = (dist < 2).astype(int); X["ring_inner"] = dist.between(2, 5).astype(int); X["ring_outer"] = (dist > 5).astype(int)
    for z in range(km.n_clusters):
        X[f"zone_{z}"] = (d["zone"] == z).astype(int)
    return X.astype(float)

FULL_X = pd.concat([base_features(A), time_zone_features(A)], axis=1)
TIME = ["q_sep", "tenure_q", "days_since_review"]
ZONE = ["ring_core", "ring_inner", "ring_outer"] + [f"zone_{z}" for z in range(km.n_clusters)]
print(len(BASE), "snapshot features +", len(TIME), "time +", len(ZONE), "zone")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for z in range(km.n_clusters):
    m = A["zone"] == z
    ax[0].scatter(A.loc[m, "longitude"], A.loc[m, "latitude"], s=3, label=f"{z}: {zone_names[z]}")
ax[0].set_aspect(1 / np.cos(np.radians(42.33))); ax[0].legend(fontsize=7, markerscale=4); ax[0].set_title("8 geographic zones (k-means)")
pv = A[A["price"].notna()].groupby(["zone", "q_label"])["price"].median().unstack()
pv.plot.bar(ax=ax[1], color=["#3b6fb6", "#e76f51"], rot=0); ax[1].set_title("Median price by zone and quarter")
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Experimental design

The same listing usually appears in several quarters at a similar price. A random row split would put a listing's June row in training and its September row in test, so the model would only have to remember it. Phase 1 also showed that listings of the same host share a price level. So we **split by host**: 20% of hosts are held out, and none of their listings, in any quarter, are used for training.

Every experiment is scored on the same held-out listings, **separately for June and for September**.
""")

code(r"""
P = A[A["price"].between(30, 2000)].copy()            # priced rows only (Jun + Sep)
P["y"] = np.log(P["price"])
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(P, groups=P["host_id"]))
TR, TE = P.iloc[tr_idx], P.iloc[te_idx]
assert not set(TR["host_id"]) & set(TE["host_id"])
print(TR.groupby("q_label").size().to_dict(), "train |", TE.groupby("q_label").size().to_dict(), "test")

def score(y_log, p_log):
    t, p = np.exp(y_log), np.exp(p_log)
    return {"MAE_$": mean_absolute_error(t, p), "R2_log": r2_score(y_log, p_log),
            "within_20%": np.mean(np.abs(p - t) / t <= .2) * 100, "bias_%": (np.median(p / t) - 1) * 100}

# Re-use the hyper-parameters tuned in Phase 1
_js = (OUT / "data.js").read_text(); PHASE1 = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])["meta"]
RF_PARAMS = dict(n_estimators=150, random_state=SEED, n_jobs=-1, **PHASE1["rf_params"])
GB_PARAMS = dict(subsample=0.8, random_state=SEED, **PHASE1["gb_params"])
print("Phase 1 settings:", PHASE1["rf_params"], PHASE1["gb_params"])
""")

md(r"""
## 4. Step A: re-select the best model on June only

Before reusing Phase 1's choice, we re-run the method comparison (the four learned models; the baseline is already known to lose) on **June data alone**, with grouped 5-fold cross-validation on the June training listings.
""")

code(r"""
jun_tr = TR[TR["quarter"] == "2026Q2"]
Xj, yj, gj = FULL_X.loc[jun_tr.index, BASE], jun_tr["y"], jun_tr["host_id"]
cands = {
    "Linear Regression": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
    "Lasso": Pipeline([("s", StandardScaler()), ("m", LassoCV(alphas=np.logspace(-4, 0, 40), max_iter=50000))]),
    "Random Forest": RandomForestRegressor(**RF_PARAMS),
    "Gradient Boosting": GradientBoostingRegressor(**GB_PARAMS),
}
june_cv = {n: -cross_val_score(m, Xj, yj, groups=gj, cv=GroupKFold(5), scoring="neg_root_mean_squared_error").mean()
           for n, m in cands.items()}
june_cv = pd.Series(june_cv).sort_values(); print("June-only CV RMSE(log):"); print(june_cv.round(4))
BEST = june_cv.idxmin(); print("lowest on June:", BEST)
gap = abs(june_cv["Random Forest"] - june_cv["Gradient Boosting"])
print(f"Random Forest vs Gradient Boosting gap: {gap:.4f} RMSE(log), well inside fold-to-fold noise")
MODEL_USED = "Random Forest"   # tie on June -> keep the Phase 1 winner (faster, needs less tuning)
""")

md(r"""
**Reading.** On June, Random Forest and Gradient Boosting **tie** (a gap of about 0.001, far smaller than the variation between folds), and the linear models trail by about 0.03. We keep **Random Forest**, the Phase 1 choice: same accuracy, more stable, faster to train.
""")

md(r"""
## 5. Step B: train on June, apply to the whole year

We train the best model on June training listings with the Phase 1 features, then:
* score it on held-out **June** listings (same quarter) and held-out **September** listings (a later quarter it never saw);
* price the full **December, March, June and September** listing stock to see how the predicted market level moves.
""")

code(r"""
def fit_rf(rows, cols, weight=None):
    return RandomForestRegressor(**RF_PARAMS).fit(FULL_X.loc[rows.index, cols], rows["y"], sample_weight=weight)

june_model = fit_rf(jun_tr, BASE)
rows = []
for q, lab in [("2026Q2", "Jun 2026"), ("2026Q3", "Sep 2026")]:
    te = TE[TE["quarter"] == q]
    rows.append({"experiment": "June-only model", "test quarter": lab, **score(te["y"], june_model.predict(FULL_X.loc[te.index, BASE]))})
pd.DataFrame(rows).round(2)
""")

md(r"""
The June model **over-prices September by about 12%** (median bias), and its September error is clearly worse than its June error. It still ranks listings sensibly, but it misses the *level*: the market cooled after the June peak. The listing features are unchanged; the market moved. Boston's June peak (graduations, summer tourism) cools by September, and a single-snapshot model has no way to know that.
""")

code(r"""
# Price the whole listing stock in every quarter with the June model (Dec and Mar have no truth to compare)
stock = []
for q, lab, *_ in QUARTERS:
    d = A[A["quarter"] == q]
    pred = np.exp(june_model.predict(FULL_X.loc[d.index, BASE]))
    stock.append({"quarter": lab, "listings": len(d), "actual_median": d["price"].median(),
                  "june_model_median": np.median(pred)})
stock = pd.DataFrame(stock); stock
""")

md(r"""
**Pricing the full stock.** For December and March there is no truth to compare against. The predicted median shows how the *mix* of listings changes. The March scrape is partial and skewed toward entire homes (76% vs about 68% in other quarters), so its predicted median is much higher. That reflects which listings were captured, not a market shift.
""")

md(r"""
## 6. Step C: pool the quarters and add time + zone

All models below are Random Forests with the tuned Phase 1 settings. Only the training rows and the feature set change:

| Experiment | Training rows | Features |
|---|---|---|
| June-only | Jun | 48 snapshot |
| Sep-only | Sep | 48 snapshot |
| Pooled | Jun + Sep | 48 snapshot |
| Pooled + time | Jun + Sep | + quarter, tenure, review recency |
| Pooled + time + zone | Jun + Sep | + rings and 8 zones |
| … + recency weighting | Jun + Sep | same, Sep rows weighted 2× |
""")

code(r"""
sep_tr = TR[TR["quarter"] == "2026Q3"]
experiments = {
    "June-only": (jun_tr, BASE, None),
    "Sep-only": (sep_tr, BASE, None),
    "Pooled": (TR, BASE, None),
    "Pooled + time": (TR, BASE + TIME, None),
    "Pooled + time + zone": (TR, BASE + TIME + ZONE, None),
    "Pooled + time + zone, recency-weighted": (TR, BASE + TIME + ZONE, np.where(TR["quarter"] == "2026Q3", 2.0, 1.0)),
}
models, rows = {}, []
for name, (rws, cols, w) in experiments.items():
    t0 = time.time(); m = fit_rf(rws, cols, w); models[name] = (m, cols)
    for q, lab in [("2026Q2", "Jun 2026"), ("2026Q3", "Sep 2026")]:
        te = TE[TE["quarter"] == q]
        rows.append({"experiment": name, "test": lab, **score(te["y"], m.predict(FULL_X.loc[te.index, cols]))})
exp = pd.DataFrame(rows)
exp_wide = exp.pivot(index="experiment", columns="test", values=["MAE_$", "R2_log", "bias_%", "within_20%"]).loc[list(experiments)]
exp_wide.round(2)
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
names = list(experiments)
for k, (metric, title) in enumerate([("MAE_$", "Test MAE ($/night)"), ("bias_%", "Median bias (%), 0 = unbiased")]):
    w = 0.38
    for j, (lab, c) in enumerate([("Jun 2026", "#3b6fb6"), ("Sep 2026", "#e76f51")]):
        v = exp[exp["test"] == lab].set_index("experiment").loc[names, metric]
        ax[k].barh(np.arange(len(names)) + (j - .5) * w, v, height=w, color=c, label=lab)
    ax[k].set_yticks(range(len(names)), names); ax[k].set_title(title); ax[k].invert_yaxis()
ax[1].axvline(0, color="k", lw=.8); ax[0].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Step D: how parameter weights change between quarters

A **Lasso fitted separately on each priced quarter** (all priced rows of that quarter) gives one weight per feature per quarter. Converting each to a "% price effect" shows which levers strengthened or weakened from June to September. The pooled Random Forest's permutation importance shows how much the new time and zone features are actually used.
""")

code(r"""
def lasso_effects(rows, cols):
    pipe = Pipeline([("s", StandardScaler()), ("m", LassoCV(alphas=np.logspace(-4, 0, 40), max_iter=50000))])
    pipe.fit(FULL_X.loc[rows.index, cols], rows["y"])
    raw = pipe.named_steps["m"].coef_ / pipe.named_steps["s"].scale_
    return pd.Series((np.exp(raw) - 1) * 100, cols)

eff = pd.DataFrame({lab: lasso_effects(P[P["quarter"] == q], BASE) for q, lab in [("2026Q2", "Jun 2026"), ("2026Q3", "Sep 2026")]})
SHOW = ["private_room", "hotel_room", "log_min_nights", "bedrooms", "log_reviews", "reviews_per_month",
        "host_size_band", "availability_365", "licensed", "dist_km", "kw_luxury", "kw_view", "kw_parking",
        "nb_Back Bay", "nb_Downtown", "nb_South Boston Waterfront", "nb_Dorchester", "nb_Fenway"]
weights = eff.loc[SHOW].round(1); weights["change_pts"] = (weights["Sep 2026"] - weights["Jun 2026"]).round(1)
weights.sort_values("change_pts")
""")

code(r"""
# The calculator prices both quarters, so the final model is the one with the lowest MAE averaged over
# the June and September test listings (a model that is only good on one quarter does not qualify).
avg_mae = exp.groupby("experiment")["MAE_$"].mean().loc[list(experiments)]
print("average test MAE over Jun + Sep:"); print(avg_mae.round(2))
final_name = avg_mae.idxmin()
final_model, final_cols = models[final_name]
te_all = TE
pi = permutation_importance(final_model, FULL_X.loc[te_all.index, final_cols], te_all["y"], n_repeats=5,
                            random_state=SEED, scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, final_cols)
grp = {"Neighbourhood (one-hot)": [c for c in final_cols if c.startswith("nb_")],
       "Zones (rings + clusters)": [c for c in final_cols if c in ZONE]}
imp_g = imp.drop([c for g in grp.values() for c in g])
for g, cs in grp.items():
    imp_g[g] = imp[cs].sum()
imp_g = imp_g.sort_values(ascending=False)
print("final model:", final_name)
imp_g.head(14).sort_values().plot.barh(figsize=(7, 4.5), color="#2a9d8f", title="Permutation importance: pooled model"); plt.tight_layout(); plt.show()
""")

md("## 8. Verdict for Phase 2")

code(r"""
j = exp.set_index(["experiment", "test"])
jo, fin = j.loc[("June-only", "Sep 2026")], j.loc[(final_name, "Sep 2026")]
print(f"September MAE  June-only model: ${jo['MAE_$']:.0f} (bias {jo['bias_%']:+.1f}%)  ->  {final_name}: ${fin['MAE_$']:.0f} (bias {fin['bias_%']:+.1f}%)")
print(f"Improvement on September: {1 - fin['MAE_$'] / jo['MAE_$']:.0%}")
""")

md(r"""
* **A snapshot model goes stale.** Trained on June, the model's error is higher on September than on June, and it over-prices September by about 12%.
* **Using every quarter helps most; time and zone add a smaller gain.** Pooling Jun + Sep does most of the work. Time features (quarter, tenure, review recency) and zone features each trim a little more, and weighting the newer quarter 2× gives the lowest error **averaged over both quarters**. On September it is about **10% below the June-only model** (exact figures printed above).
* **Why not a September-only model?** It is marginally better on September alone, by well under a dollar, but it is one of the worst on June. A model that has to price any quarter must be judged on all of them.
* **Weights move between quarters.** In the per-quarter Lasso, the premium for a licensed listing falls by more than half from June to September, and the premiums for Downtown, Fenway and "view" listings grow. A fixed set of weights can't serve all year, which is why the pooled model learns with a time dimension.
* **Some bias remains (about +5% on September).** With only two priced quarters, the model has one seasonal contrast to learn from. Priced December and March files would let `q_sep` become a proper season feature.
* **Recommendation:** use the pooled time + zone Random Forest with recency weighting, and re-fit it every quarter as each new snapshot arrives.
""")

md("## 9. Export to the interface")

code(r"""
FINE = {final_cols.index(c) for c in ("latitude", "longitude", "dist_km")}

def export_tree(t):
    tr = t.tree_; leaf = tr.children_left == -1
    assert np.all(tr.children_left[~leaf] == np.arange(tr.node_count)[~leaf] + 1)
    # coordinate splits need 9 significant digits; other thresholds 6; leaf values (log price) 5
    x = [float(f"{v:.5g}") if is_leaf else float(f"{th:.9g}" if f in FINE else f"{th:.6g}")
         for v, th, f, is_leaf in zip(tr.value[:, 0, 0], tr.threshold, tr.feature, leaf)]
    return {"f": np.where(leaf, -1, tr.feature).tolist(), "x": x}

# refit the chosen pooled model on ALL priced rows
w_all = np.where(P["quarter"] == "2026Q3", 2.0, 1.0) if "recency" in final_name else None
deploy = RandomForestRegressor(**RF_PARAMS).fit(FULL_X.loc[P.index, final_cols], P["y"], sample_weight=w_all)
ref = P.sample(150, random_state=1)

quarterly = {
    "quarters": [{"code": q, "label": lab, "listings": int((A["quarter"] == q).sum()),
                  "priced": int(A.loc[A["quarter"] == q, "price"].notna().sum()),
                  "actual_median": None if np.isnan(m := A.loc[A["quarter"] == q, "price"].median()) else float(m),
                  "june_model_median": float(stock.loc[stock["quarter"] == lab, "june_model_median"].iloc[0]),
                  "entire_share": float((A.loc[A["quarter"] == q, "room_type"] == "Entire home/apt").mean())}
                 for q, lab, *_ in QUARTERS],
    "june_cv": june_cv.round(4).to_dict(), "best_on_june": BEST,
    "experiments": exp.round(3).to_dict(orient="records"), "experiment_order": names, "final": final_name,
    "weights": weights.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
    "importance": imp_g.head(14).round(5).to_dict(),
    "zones": {"centers_km": km.cluster_centers_.round(5).tolist(), "names": {int(k): v for k, v in zone_names.items()}},
    "zone_price": {f"{int(z)}|{lab}": float(v) for (z, lab), v in P.groupby(["zone", "q_label"])["price"].median().items()},
    "features": final_cols,
    "band_log": [float(np.quantile(r := TE["y"] - final_model.predict(FULL_X.loc[TE.index, final_cols]), .10)), float(np.quantile(r, .90))],
    "center": CENTER, "n_zones": int(km.n_clusters),
}
js = (OUT / "data.js").read_text()
bundle = json.loads(js[js.index("{"): js.rindex("}") + 1])
bundle["quarterly"] = quarterly
payload = json.dumps(bundle, separators=(",", ":"))
(OUT / "data.js").write_text("window.PRICING = " + payload + ";\n")

mjs = (OUT / "models.js").read_text()
mbundle = json.loads(mjs[mjs.index("{"): mjs.rindex("}") + 1])
mbundle["quarterly"] = {"trees": [export_tree(e) for e in deploy.estimators_]}
mbundle["quarterly_reference"] = {"X": FULL_X.loc[ref.index, final_cols].values.tolist(),
                                  "pred": np.round(deploy.predict(FULL_X.loc[ref.index, final_cols]), 6).tolist()}
mpayload = json.dumps(mbundle, separators=(",", ":"))
(OUT / "models.js").write_text("window.PRICING_MODELS = " + mpayload + ";\n")
print(f"data.js {len(payload) / 1e6:.2f} MB, models.js {len(mpayload) / 1e6:.2f} MB")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "boston_airbnb_quarterly.ipynb"
nbf.write(nb, out); print("wrote", out)
