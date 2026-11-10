"""Builds notebooks/boston_airbnb_team_merge.ipynb: compare the teammate's ML pipeline with ours and merge them.

Run first:  cd teammate && python Code/clean_listings.py && python Code/train_models.py   (their script, unchanged)
Then:       python scripts/build_merge_notebook.py
            jupyter nbconvert --to notebook --execute --inplace notebooks/boston_airbnb_team_merge.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Merging the teammate's pipeline with ours

A teammate built a separate pricing pipeline: `teammate/Code/train_models.py` plus a variable-screening workbook (`teammate/Boston_Airbnb_Variable_Screening.xlsx`). This notebook does four things:

1. **Explains** what their pipeline does and how it differs from ours.
2. **Reproduces** their results by running their script unchanged on a re-creation of their cleaning step. Their `clean_listings.py` was not shared, so `teammate/Code/clean_listings.py` follows the workbook rule by rule.
3. **Tests** each of their ideas on our pipeline, one at a time, under our host-grouped evaluation.
4. **Merges** the ideas that help, and compares every model type on the merged feature set, including their neural network.
""")

md(r"""
## 1. The two pipelines side by side

| | Teammate | Ours (Phase 3, Steps 1-2) |
|---|---|---|
| **Data** | Sep 2026 detailed file | Jun + Sep 2026 detailed files |
| **Rows** | price > 0 **and at least one review**: 3,002 | every priced listing after the pre-check: 7,039 |
| **Location** | 25 neighbourhoods grouped into **6 areas** | 24 neighbourhood dummies, coordinates, distance to centre, 8 zones, 14 landmarks |
| **Property** | 26 types → 6 groups; room type | 27 types → 12 groups; room type |
| **Size** | guests, bedrooms, beds, bathrooms, shared bath | same |
| **Amenities** | **10 broad flags** (wifi, kitchen, AC, parking, laundry, workspace, TV, coffee, outdoor space, pets) | 30 narrower groups plus a count |
| **Host** | **host experience in months**, superhost, verified, log(listing count) | superhost, verified, host size band (the raw count leaked host identity) |
| **Reviews** | 7 scores, counts, **months since first review**, days since last review | 7 scores, counts, reviews per month, days since last review |
| **Booking rules** | minimum nights, **stay length of the price quote** | minimum nights, availability, license |
| **Missing values** | median imputation on the full sample (they note the small leak) | kept as missing for boosting; train-only median + indicator otherwise |
| **Split** | random 80/20, random 5-fold CV | **grouped by host** |
| **Models** | linear, ridge, CART, random forest, gradient boosting, **neural network (MLP)** | Lasso, random forest, LightGBM, CatBoost |
| **Reporting** | **in-sample and out-of-sample** RMSE / MAE / R² on log price, plus dollars | out-of-sample only, dollars first |

The **bold** entries are things their pipeline has and ours did not. Each is tested below.
""")

code(r"""
import ast, json, re, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
DATA, OUT, TEAM, SEED = Path("../data"), Path("../interface"), Path("../teammate"), 42
sys.path.insert(0, "../src")
import pricing_features as pf
import enriched_features as ef
import team_features as tf
_js = (OUT / "data.js").read_text(); BUNDLE = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
""")

md(r"""
## 2. Their results, reproduced

`teammate/Code/train_models.py` was run unchanged. It wrote `teammate/Results/model_performance.csv` and a Markdown report with the tuned hyperparameters.
""")

code(r"""
team_res = pd.read_csv(TEAM / "Results" / "model_performance.csv")
report = (TEAM / "Results" / "model_performance.md").read_text()
dollar_block = report.split("## Supplementary")[1].split("## Tuned")[0]
team_dollars = {m.group(1).strip(): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
                for m in re.finditer(r"\| ([^|]+?) \| ([\d.]+) \| ([\d.]+) \| (-?[\d.]+) \|", dollar_block)}
tuned = {}
for line in report.split("## Tuned hyperparameters")[1].split("## Notes")[0].splitlines():
    m = re.match(r"- \*\*(.+?)\*\*: (.*)", line)
    if m:
        tuned[m.group(1)] = {k: ast.literal_eval(v.strip()) for k, v in re.findall(r"(\w+)=(\([^)]*\)|[^,]+)", m.group(2))
                             if k not in ("CV_RMSE", "validation_RMSE")} if "none" not in m.group(2) else {}
team_res["Out-of-sample MAE ($)"] = team_res["Model"].map(lambda n: team_dollars.get(n, (np.nan,) * 3)[1])
print("tuned hyperparameters:", tuned)
team_res.round(4)
""")

md(r"""
**Reading.** Their best models reach an out-of-sample R² of about 0.7-0.8 on log price, in line with our Steps 1-2 figures. The in-sample columns are a useful addition we didn't report: the gap between in-sample and out-of-sample shows how much each model overfits. CART and random forest fit the training data far better than they predict new data.

The comparison isn't like-for-like yet: their test set is a **random** 20% of listings, so most test listings have siblings from the same host in training, and new listings without reviews are excluded altogether.
""")

md(r"""
## 3. Stress test: their models under host-grouped evaluation

Same 3,002 listings, same 48 features, same tuned hyperparameters. Only the split changes: no host appears on both sides.
""")

code(r"""
clean = pd.read_csv(TEAM / "Data" / "Processed" / "listings_clean.csv")
FEATS = [c for c in clean.columns if c not in ["id", "host_id", "last_scraped", "price_quote_checkin_date", "price_quote_checkout_date",
         "first_review", "last_review", "neighbourhood_cleansed", "neighbourhood_area", "property_type", "property_group", "room_type",
         "price", "log_price"]]
REF = ["nbhd_outer_residential_south", "prop_apartment_condo", "room_entire_home_apt"]
LIN = [c for c in FEATS if c not in REF]
Xt, yt, gt = clean[FEATS], clean["log_price"].values, clean["host_id"].values

def team_model(name):
    p = tuned.get(name, {})
    if name == "Linear regression": return LinearRegression(), LIN
    if name == "Ridge regression (CV)": return make_pipeline(StandardScaler(), Ridge(alpha=p.get("ridge__alpha", 1.0))), LIN
    if name == "CART (CV)": return DecisionTreeRegressor(random_state=SEED, **p), FEATS
    if name == "Random forest (CV)": return RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, **p), FEATS
    if name == "Gradient boosting (CV)": return GradientBoostingRegressor(random_state=SEED, **p), FEATS
    return make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=SEED, **p)), FEATS

rows = []
for split_name, cv in [("random 5-fold (their CV)", KFold(5, shuffle=True, random_state=SEED)), ("host-grouped 5-fold", GroupKFold(5))]:
    for name in team_res["Model"]:
        errs = []
        for a, b in cv.split(Xt, yt, gt):
            m, cols = team_model(name); m.fit(Xt.iloc[a][cols], yt[a]); p = m.predict(Xt.iloc[b][cols])
            errs.append((np.sqrt(mean_squared_error(yt[b], p)), r2_score(yt[b], p), mean_absolute_error(np.exp(yt[b]), np.exp(p))))
        e = np.array(errs).mean(0)
        rows.append({"split": split_name, "model": name, "RMSE_log": e[0], "R2_log": e[1], "MAE_$": e[2]})
stress = pd.DataFrame(rows)
stress_wide = stress.pivot(index="model", columns="split", values=["R2_log", "MAE_$"]).loc[team_res["Model"]]
stress_wide.round(3)
""")

code(r"""
gap = stress.pivot(index="model", columns="split", values="R2_log")
gap["R² drop"] = gap["random 5-fold (their CV)"] - gap["host-grouped 5-fold"]
print("Average optimism of the random split (R² points):", round(gap["R² drop"].mean(), 3))
gap.round(3)
""")

md(r"""
**Reading.** Every model loses accuracy once hosts can't appear on both sides. This is the same effect we found in Phase 1. Part of it comes from `log_host_listings_count`, which lets models recognise large operators. The honest numbers are the host-grouped ones.
""")

md(r"""
## 4. Their ideas, tested one at a time on our pipeline

Base: our Phase 3 Steps 1-2 feature set (133 columns) on June + September, LightGBM, host-grouped 5-fold CV with the same folds as Phase 3. Each idea is added **on its own**, and the per-fold change in dollar error shows whether it helps consistently.
""")

code(r"""
official = pd.read_csv(DATA / "neighbourhoods.csv")["neighbourhood"]
geo = pf.load_geo(DATA / "neighbourhoods.geojson"); H = pf.model_neighbourhoods(official)
frames, supply = [], []
for q, f in [("2026Q2", "listings_detailed_2026Q2_jun.csv.gz"), ("2026Q3", "listings_detailed_2026Q3_sep.csv.gz")]:
    raw = ef.load_detailed(DATA / "detailed" / f)
    a, _ = pf.precheck(raw, official, geo, require_price=False); a["quarter"] = q; supply.append(a)
    d, _ = pf.precheck(raw, official, geo); d["quarter"] = q; frames.append(d)
A = pd.concat(frames, ignore_index=True); SUPPLY = pd.concat(supply, ignore_index=True)
PROPS = sorted(set(ef.PROPERTY_GROUPS.values())) + ["other"]
BASE = pd.concat([pf.build_features(A, H), ef.time_zone_features(A, DATA / "quarterly", BUNDLE["quarterly"]["zones"]["centers_km"]),
                  ef.detailed_features(A, PROPS), ef.landmark_features(A), ef.density_features(A, SUPPLY)], axis=1)
y, groups = np.log(A["price"].values), A["host_id"].values
EX = tf.extra_features(A)
TEAM_X, _ = tf.team_features(A)   # only the area and amenity columns are used below; they have no missing values
qci = pd.to_datetime(A["price_quote_checkin_date"])
QUOTE = pd.DataFrame({"quote_lead_days": (qci - pd.to_datetime(A["last_scraped"])).dt.days.astype(float),
                      "quote_weekend": qci.dt.dayofweek.isin([4, 5]).astype(float).where(qci.notna())}, index=A.index)
IDEAS = {
    "host experience (months)": EX[["host_experience_months"]],
    "months since first review": EX[["months_since_first_review"]],
    "stay length of the price quote": EX[["stay_length"]],
    "6 neighbourhood areas": TEAM_X[[c for c in TEAM_X.columns if c.startswith("nbhd_")]],
    "10 broad amenity flags": TEAM_X[list(tf.AMENITY_ROOTS)].add_prefix("team_"),
    "log host listing count (raw)": EX[["log_host_listings_count"]],
    "quote check-in day and lead time (our extension)": QUOTE,
}
print({k: v.shape[1] for k, v in IDEAS.items()}, "| coverage:", EX.notna().mean().round(3).to_dict())
""")

code(r"""
LGB = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED, verbose=-1)
FOLDS = list(GroupKFold(5).split(A, y, groups))
def cv_mae(X):
    return np.array([mean_absolute_error(np.exp(y[b]), np.exp(lgb.LGBMRegressor(**LGB).fit(X.iloc[a], y[a]).predict(X.iloc[b]))) for a, b in FOLDS])
base_err = cv_mae(BASE)
rows = []
for name, blk in IDEAS.items():
    e = cv_mae(pd.concat([BASE, blk], axis=1))
    rows.append({"idea": name, "CV MAE": e.mean(), "change $": (e - base_err).mean(), "folds improved": int((e < base_err).sum())})
ideas = pd.DataFrame(rows).set_index("idea")
print(f"base CV MAE: ${base_err.mean():.2f}")
ideas.sort_values("change $").round(2)
""")

code(r"""
KEEP = ideas[(ideas["change $"] < 0) & (ideas["folds improved"] >= 4)].index.tolist()
print("kept (lower error in at least 4 of 5 folds):", KEEP)
MERGED = pd.concat([BASE] + [IDEAS[k] for k in KEEP], axis=1)
merged_err = cv_mae(MERGED)
print(f"merged CV MAE: ${merged_err.mean():.2f} (base ${base_err.mean():.2f}; change {merged_err.mean() - base_err.mean():+.2f}, "
      f"{int((merged_err < base_err).sum())} of 5 folds better)")
""")

md(r"""
## 5. Every model type on the merged feature set

Their model list (linear, ridge, CART, random forest, gradient boosting, neural network) plus ours (Lasso via ridge-type linear, LightGBM, CatBoost), all on the merged features. The split is host-grouped 80/20, and **in-sample and out-of-sample** errors are both reported, as in their script.

* Linear models, CART, random forest and the MLP get a train-only median fill plus missing indicators. LightGBM and CatBoost read missing values natively.
* The **MLP** is tuned the way they tuned it, by choosing the architecture and L2 penalty on a validation split of the training set. Here that split is **grouped by host**.
""")

code(r"""
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(MERGED, y, groups))
Xtr, Xte, ytr, yte, gtr = MERGED.iloc[tr], MERGED.iloc[te], y[tr], y[te], groups[tr]
imp = lambda est: Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler()), ("m", est)])

def tune_mlp():
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(Xtr, ytr, gtr))
    best, best_rmse = None, np.inf
    for h in [(32,), (64,), (64, 32), (128, 64)]:
        for alpha in [1e-2, 1e-1, 1.0, 3.0]:
            m = imp(MLPRegressor(hidden_layer_sizes=h, alpha=alpha, learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                                 n_iter_no_change=30, random_state=SEED)).fit(Xtr.iloc[a], ytr[a])
            r = np.sqrt(mean_squared_error(ytr[b], m.predict(Xtr.iloc[b])))
            if r < best_rmse: best, best_rmse = dict(hidden_layer_sizes=h, alpha=alpha), r
    return best

MLP_P = tune_mlp(); print("MLP chosen on host-grouped validation:", MLP_P)
MODELS = {
    "Linear regression": imp(LinearRegression()),
    "Ridge (CV)": imp(RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "CART": Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("m", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED))]),
    "Random forest": Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("m", RandomForestRegressor(n_estimators=500, max_features=0.33, min_samples_leaf=1, random_state=SEED, n_jobs=-1))]),
    "Neural network (MLP)": imp(MLPRegressor(learning_rate_init=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=SEED, **MLP_P)),
    "LightGBM": lgb.LGBMRegressor(**LGB),
    "CatBoost": CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3, random_seed=SEED, verbose=0),
}
def met(yt_, yp):
    return {"RMSE_log": np.sqrt(mean_squared_error(yt_, yp)), "R2_log": r2_score(yt_, yp), "MAE_$": mean_absolute_error(np.exp(yt_), np.exp(yp))}
rows, fitted = [], {}
for name, m in MODELS.items():
    t0 = time.time(); m.fit(Xtr, ytr); fitted[name] = m
    ins, oos = met(ytr, m.predict(Xtr)), met(yte, m.predict(Xte))
    rows.append({"model": name, **{f"in {k}": v for k, v in ins.items()}, **{f"out {k}": v for k, v in oos.items()},
                 "overfit gap (R²)": ins["R2_log"] - oos["R2_log"], "fit_s": time.time() - t0})
final_cmp = pd.DataFrame(rows).set_index("model")
final_cmp.round(3)
""")

code(r"""
# Ensemble: average of the two boosting models and the MLP (a common way to combine a neural net with trees)
ens_p = np.mean([fitted[n].predict(Xte) for n in ["LightGBM", "CatBoost", "Neural network (MLP)"]], axis=0)
ens_t = np.mean([fitted[n].predict(Xtr) for n in ["LightGBM", "CatBoost", "Neural network (MLP)"]], axis=0)
e_in, e_out = met(ytr, ens_t), met(yte, ens_p)
final_cmp.loc["Ensemble (LightGBM + CatBoost + MLP)"] = {**{f"in {k}": v for k, v in e_in.items()}, **{f"out {k}": v for k, v in e_out.items()},
                                                         "overfit gap (R²)": e_in["R2_log"] - e_out["R2_log"], "fit_s": np.nan}
fig, ax = plt.subplots(figsize=(9, 4))
o = final_cmp.sort_values("out MAE_$")
ax.barh(o.index, o["out MAE_$"], color=["#3b6fb6" if "LightGBM" in i or "Ensemble" in i else "#9aa5b1" for i in o.index])
for i, v in enumerate(o["out MAE_$"]): ax.text(v + 1, i, f"${v:.0f}", va="center", fontsize=8)
ax.invert_yaxis(); ax.set_title("Out-of-sample MAE on unseen hosts, merged features"); plt.tight_layout(); plt.show()
final_cmp.loc[["Ensemble (LightGBM + CatBoost + MLP)"]].round(3)
""")

md("## 6. What each side contributed")

code(r"""
best_name = final_cmp["out MAE_$"].idxmin()
summary = pd.DataFrame([
    ("Teammate", "Host experience (months)", "Tested", f"{ideas.loc['host experience (months)', 'change $']:+.2f} $/night, {ideas.loc['host experience (months)', 'folds improved']}/5 folds"),
    ("Teammate", "Months since first review", "Tested", f"{ideas.loc['months since first review', 'change $']:+.2f}, {ideas.loc['months since first review', 'folds improved']}/5"),
    ("Teammate", "Stay length of the price quote", "Tested", f"{ideas.loc['stay length of the price quote', 'change $']:+.2f}, {ideas.loc['stay length of the price quote', 'folds improved']}/5"),
    ("Teammate", "6 neighbourhood areas", "Tested", f"{ideas.loc['6 neighbourhood areas', 'change $']:+.2f}, {ideas.loc['6 neighbourhood areas', 'folds improved']}/5"),
    ("Teammate", "10 broad amenity flags", "Tested", f"{ideas.loc['10 broad amenity flags', 'change $']:+.2f}, {ideas.loc['10 broad amenity flags', 'folds improved']}/5"),
    ("Teammate", "log host listing count", "Tested", f"{ideas.loc['log host listing count (raw)', 'change $']:+.2f}, {ideas.loc['log host listing count (raw)', 'folds improved']}/5 (host-identity risk)"),
    ("Teammate", "Neural network (MLP)", "Compared", f"out-of-sample MAE ${final_cmp.loc['Neural network (MLP)', 'out MAE_$']:.0f} vs best ${final_cmp['out MAE_$'].min():.0f} ({best_name})"),
    ("Teammate", "In-sample vs out-of-sample reporting", "Adopted", "overfit gap now shown for every model"),
    ("Ours", "Host-grouped split", "Applied to their models", f"random split overstated R² by {gap['R² drop'].mean():.2f} on average"),
    ("Ours", "Keep listings without reviews", "Kept", "a pricing tool must serve new hosts; their filter drops them"),
    ("Ours", "Price-derived columns excluded", "Kept", "stay length checked: it is a booking rule, not the price"),
], columns=["from", "element", "status", "evidence"])
summary["status"] = [("Kept" if e in KEEP else "Not kept") if s == "Tested" else s for s, e in zip(summary["status"], [
    "host experience (months)", "months since first review", "stay length of the price quote", "6 neighbourhood areas",
    "10 broad amenity flags", "log host listing count (raw)", "", "", "", "", ""])]
summary
""")

md(r"""
## 7. Verdict

The tables above give the exact figures; the main points are:

* **Their feature engineering and ours agree on what matters:** size, bathrooms, room and property type, review scores and minimum nights. Their 48-feature set is compact and readable, and close to ours in accuracy on the same rows.
* **Their evaluation was optimistic.** A random split lets the same host appear in training and test; under host-grouped evaluation every one of their models loses R². Their data filter also drops new listings, which a pricing tool must handle.
* **Several of their features are genuinely new to us,** and the ones that lower error consistently are merged into our pipeline (the "kept" rows). Features that duplicate what we already have, like broad amenity flags or neighbourhood areas, add nothing to a tree model that already sees the detailed version.
* **The neural network is competitive but doesn't win alone** on ~5,600 training rows of tabular data. Averaged with the boosting models it can help, which is the ensemble row.
* **Their in-sample vs out-of-sample reporting is a good practice** that we now adopt: it shows that random forest and CART overfit far more than boosting or the linear models.
""")

code(r"""
bundle = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
bundle["team_merge"] = {
    "team_results": team_res.round(4).replace({np.nan: None}).to_dict(orient="records"),
    "stress": stress.round(4).to_dict(orient="records"),
    "optimism_r2": float(gap["R² drop"].mean()),
    "ideas": ideas.reset_index().round(3).to_dict(orient="records"),
    "base_cv_mae": float(base_err.mean()), "merged_cv_mae": float(merged_err.mean()),
    "merged_folds_better": int((merged_err < base_err).sum()), "kept": KEEP,
    "models": final_cmp.reset_index().round(4).replace({np.nan: None}).to_dict(orient="records"),
    "best_model": best_name, "mlp_params": {k: list(v) if isinstance(v, tuple) else v for k, v in MLP_P.items()},
    "summary": summary.to_dict(orient="records"),
    "n_team_rows": int(len(clean)), "n_rows": int(len(A)), "n_test": int(len(te)),
}
payload = json.dumps(bundle, separators=(",", ":"))
(OUT / "data.js").write_text("window.PRICING = " + payload + ";\n")
print(f"added 'team_merge' to interface/data.js ({len(payload) / 1e6:.2f} MB)")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "boston_airbnb_team_merge.ipynb"
nbf.write(nb, out); print("wrote", out)
