"""Builds notebooks/boston_airbnb_overfitting_check.ipynb: how likely is the merged model to be overfit,
what errors could it make, and what should come next (including which deep-learning route).

Run:  python scripts/build_overfit_notebook.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/boston_airbnb_overfitting_check.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# How much should we trust the merged model?

After merging the teammate's pipeline, the recommended model is **LightGBM on 141 features**: Phase 3 Steps 1-2, plus months since the first review and the quoted night's day of week. Its host-grouped CV error is about $85 a night. Before building more on it, this notebook asks how that number could be wrong.

| Risk | How it could happen here | Check |
|---|---|---|
| Model overfitting | Trees memorise training listings | §1 in-sample vs held-out gap, §2 learning curves |
| Selection overfitting | We tried many feature ideas on the same folds and kept the winners | §3 noise-feature test, §4 nested selection |
| Time drift | Trained on summer prices, used in another season | §5 June → September test |
| Over-confident ranges | The "80% range" may cover less than 80% | §5 interval coverage |
| Fragile tuning | Results depend on one lucky hyperparameter setting | §6 sensitivity |
| Systematic errors | Some kinds of listing are always over- or under-priced | §7 error analysis |

Every check keeps hosts apart between training and evaluation, as in all earlier phases.
""")

code(r"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
DATA, OUT, SEED = Path("../data"), Path("../interface"), 42
sys.path.insert(0, "../src")
import enriched_features as ef
import team_features as tf
_js = (OUT / "data.js").read_text(); BUNDLE = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
A, X, y, groups = ef.build_merged_matrix(DATA, BUNDLE["quarterly"]["zones"]["centers_km"])
KEPT = ["months_since_first_review"] + [c for c in X.columns if c.startswith("quote_dow_")]
BASE = X.drop(columns=KEPT)
LGB = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED, verbose=-1)
FOLDS = list(GroupKFold(5).split(X, y, groups))
mae = lambda yt, yp: mean_absolute_error(np.exp(yt), np.exp(yp))
print(X.shape, "| base without the merged ideas:", BASE.shape[1], "features")
""")

md(r"""
## 1. In-sample vs held-out, every fold

R² on the rows a model was trained on, against R² on the held-out hosts. A large gap means the model memorises. It is only a *problem* if the held-out score is worse than a simpler model's.
""")

code(r"""
imp = lambda est: Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler()), ("m", est)])
MODELS = {
    "Ridge": lambda: imp(RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "Random forest": lambda: Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                                       ("m", RandomForestRegressor(n_estimators=300, max_features=0.33, random_state=SEED, n_jobs=-1))]),
    "Neural network (MLP)": lambda: imp(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=3.0, max_iter=2000, early_stopping=True,
                                                     n_iter_no_change=30, random_state=SEED)),
    "LightGBM": lambda: lgb.LGBMRegressor(**LGB),
}
rows = []
for name, make in MODELS.items():
    for k, (a, b) in enumerate(FOLDS):
        m = make().fit(X.iloc[a], y[a])
        rows.append({"model": name, "fold": k, "train_R2": r2_score(y[a], m.predict(X.iloc[a])),
                     "heldout_R2": r2_score(y[b], m.predict(X.iloc[b])), "heldout_MAE": mae(y[b], m.predict(X.iloc[b]))})
gapdf = pd.DataFrame(rows)
gap = gapdf.groupby("model")[["train_R2", "heldout_R2", "heldout_MAE"]].agg(["mean", "std"]).round(3)
gap[("gap", "R2")] = (gap[("train_R2", "mean")] - gap[("heldout_R2", "mean")]).round(3)
gap.sort_values(("heldout_MAE", "mean"))
""")

md(r"""
## 2. Learning curves: would more data help, and which model would gain most?

Each model is trained on 20% to 100% of every fold's training hosts and scored on the same held-out hosts. A curve that is still falling at 100% means more data would help. This matters for deep learning: neural networks usually need the most data.
""")

code(r"""
fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
lc = []
rng = np.random.default_rng(SEED)
for k, (a, b) in enumerate(FOLDS):
    hosts = np.unique(groups[a]); rng.shuffle(hosts)
    for f in fracs:
        keep_h = set(hosts[: max(1, int(len(hosts) * f))])
        aa = a[np.isin(groups[a], list(keep_h))]
        for name, make in MODELS.items():
            m = make().fit(X.iloc[aa], y[aa])
            lc.append({"model": name, "fold": k, "frac": f, "n_train": len(aa), "MAE": mae(y[b], m.predict(X.iloc[b]))})
lc = pd.DataFrame(lc)
curve = lc.groupby(["model", "frac"]).agg(n_train=("n_train", "mean"), MAE=("MAE", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, g in curve.groupby("model"):
    ax.plot(g["n_train"], g["MAE"], marker="o", label=name)
ax.set_xlabel("training listings"); ax.set_ylabel("held-out MAE ($/night)"); ax.set_title("Learning curves (host-grouped 5-fold)"); ax.legend()
plt.tight_layout(); plt.show()
slope = curve.pivot(index="model", columns="frac", values="MAE")
slope["gain from last 20% of data ($)"] = slope[0.8] - slope[1.0]
slope["gain 20%→100% ($)"] = slope[0.2] - slope[1.0]
slope.round(2)
""")

md(r"""
## 3. How often would our "keep" rule accept a useless feature?

Every idea in the merge was judged by the same rule: lower error, in at least 4 of 5 folds. To estimate the rule's false-positive rate, 30 features that **cannot** carry information are tested the same way: 10 random numbers, 10 random yes/no flags, and 10 real features shuffled across listings.
""")

code(r"""
def cv_mae(Xm):
    return np.array([mae(y[b], lgb.LGBMRegressor(**LGB).fit(Xm.iloc[a], y[a]).predict(Xm.iloc[b])) for a, b in FOLDS])
base_err = cv_mae(BASE)
rng = np.random.default_rng(7)
real_cols = ["accommodates", "rs_location", "d_logan_airport", "am_count", "beds", "rs_value", "supply_1km", "desc_len", "baths", "rs_rating"]
noise = {}
for i in range(10): noise[f"gaussian_{i}"] = rng.normal(size=len(X))
for i in range(10): noise[f"binary_{i}"] = (rng.random(len(X)) < 0.1).astype(float)
for c in real_cols: noise[f"shuffled_{c}"] = rng.permutation(X[c].values)
nrows = []
for name, v in noise.items():
    e = cv_mae(pd.concat([BASE, pd.Series(v, index=X.index, name="noise")], axis=1))
    nrows.append({"noise feature": name, "change $": (e - base_err).mean(), "folds improved": int((e < base_err).sum())})
noise_df = pd.DataFrame(nrows)
noise_df["would be kept"] = (noise_df["change $"] < 0) & (noise_df["folds improved"] >= 4)
fp_rate = noise_df["would be kept"].mean()
print(f"base CV MAE ${base_err.mean():.2f} | noise features passing the keep rule: {noise_df['would be kept'].sum()} of {len(noise_df)} ({fp_rate:.0%})")
print(f"largest improvement from pure noise: ${-noise_df['change $'].min():.2f}; typical size of a noise change: ±${noise_df['change $'].abs().median():.2f}")
noise_df.sort_values("change $").head(8).round(3)
""")

md(r"""
## 4. Nested selection: an unbiased estimate of what the merge added

In the merge, the ideas were chosen and scored on the **same** folds, which flatters the result. Here the choice is made *inside* each training fold, with a 4-fold host-grouped split of the training hosts only. The outer held-out hosts are used only once, to score whatever was chosen.
""")

code(r"""
EX = tf.extra_features(A)
TEAM_X, _ = tf.team_features(A)
qci = pd.to_datetime(A["price_quote_checkin_date"])
IDEAS = {
    "host experience": EX[["host_experience_months"]],
    "months since first review": X[["months_since_first_review"]],
    "stay length": EX[["stay_length"]],
    "6 neighbourhood areas": TEAM_X[[c for c in TEAM_X.columns if c.startswith("nbhd_")]],
    "10 broad amenity flags": TEAM_X[list(tf.AMENITY_ROOTS)].add_prefix("team_"),
    "quoted night day of week": X[[c for c in X.columns if c.startswith("quote_dow_")]],
}
nested = []
for k, (a, b) in enumerate(FOLDS):
    inner = list(GroupKFold(4).split(a, y[a], groups[a]))
    def inner_err(Xm):
        Xa = Xm.iloc[a]
        return np.array([mae(y[a][ib], lgb.LGBMRegressor(**LGB).fit(Xa.iloc[ia], y[a][ia]).predict(Xa.iloc[ib])) for ia, ib in inner])
    e0 = inner_err(BASE)
    chosen = [n for n, blk in IDEAS.items() if (lambda e: (e - e0).mean() < 0 and (e < e0).sum() >= 3)(inner_err(pd.concat([BASE, blk], axis=1)))]
    Xsel = pd.concat([BASE] + [IDEAS[n] for n in chosen], axis=1)
    out_base = mae(y[b], lgb.LGBMRegressor(**LGB).fit(BASE.iloc[a], y[a]).predict(BASE.iloc[b]))
    out_sel = mae(y[b], lgb.LGBMRegressor(**LGB).fit(Xsel.iloc[a], y[a]).predict(Xsel.iloc[b]))
    nested.append({"outer fold": k, "ideas chosen inside the fold": ", ".join(chosen) or "none", "base MAE": out_base, "selected MAE": out_sel, "change $": out_sel - out_base})
nested = pd.DataFrame(nested)
merged_err = cv_mae(X)
print(f"non-nested estimate (merge notebook style): {merged_err.mean() - base_err.mean():+.2f} $/night")
print(f"nested (unbiased) estimate:                 {nested['change $'].mean():+.2f} $/night, better in {(nested['change $'] < 0).sum()} of 5 outer folds")
nested.round(2)
""")

md(r"""
## 5. Out-of-time test and price-range coverage

The real use is pricing *future* nights. Train on **June** listings of 80% of hosts; test on **September** listings of the other 20%: new hosts and a later season. The 80% price range is built from the model's errors on June training data (out-of-fold), then checked on September: does it contain 80% of actual prices?
""")

code(r"""
tr_h, te_h = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
train_hosts, test_hosts = set(groups[tr_h]), set(groups[te_h])
jun = np.where((A["quarter"] == "2026Q2").values & np.isin(groups, list(train_hosts)))[0]
sep_tr = np.where((A["quarter"] == "2026Q3").values & np.isin(groups, list(train_hosts)))[0]
sep_te = np.where((A["quarter"] == "2026Q3").values & np.isin(groups, list(test_hosts)))[0]
def band(idx):
    oof = np.zeros(len(idx))
    for ia, ib in GroupKFold(5).split(idx, y[idx], groups[idx]):
        oof[ib] = lgb.LGBMRegressor(**LGB).fit(X.iloc[idx[ia]], y[idx[ia]]).predict(X.iloc[idx[ib]])
    r = y[idx] - oof
    return np.quantile(r, 0.10), np.quantile(r, 0.90)
rows = []
for label, train_idx in [("trained on June only", jun), ("trained on June + September", np.concatenate([jun, sep_tr]))]:
    m = lgb.LGBMRegressor(**LGB).fit(X.iloc[train_idx], y[train_idx]); p = m.predict(X.iloc[sep_te])
    lo, hi = band(train_idx)
    cover = np.mean((y[sep_te] >= p + lo) & (y[sep_te] <= p + hi))
    rows.append({"model": label, "Sep test MAE $": mae(y[sep_te], p), "median bias %": (np.median(np.exp(p - y[sep_te])) - 1) * 100,
                 "80% range covers": cover * 100, "range width (x price)": np.exp(hi - lo)})
oot = pd.DataFrame(rows).set_index("model")
print(len(sep_te), "September test listings from unseen hosts")
oot.round(2)
""")

md(r"""
## 6. Does the result hinge on one hyperparameter setting?
""")

code(r"""
sens = []
for leaves in [15, 31, 63]:
    for mcs in [10, 20, 50]:
        for lr, n in [(0.03, 900), (0.015, 1800)]:
            p = {**LGB, "num_leaves": leaves, "min_child_samples": mcs, "learning_rate": lr, "n_estimators": n}
            e = [mae(y[b], lgb.LGBMRegressor(**p).fit(X.iloc[a], y[a]).predict(X.iloc[b])) for a, b in FOLDS]
            sens.append({"num_leaves": leaves, "min_child_samples": mcs, "learning_rate": lr, "CV MAE": np.mean(e)})
sens = pd.DataFrame(sens)
print(f"18 settings: CV MAE from ${sens['CV MAE'].min():.2f} to ${sens['CV MAE'].max():.2f} (spread ${sens['CV MAE'].max() - sens['CV MAE'].min():.2f}); "
      f"current setting ${sens.query('num_leaves == 31 and min_child_samples == 20 and learning_rate == 0.03')['CV MAE'].iloc[0]:.2f}")
sens.sort_values("CV MAE").head(5).round(2)
""")

md(r"""
## 7. Where the model goes wrong

Out-of-fold predictions for every listing (host-grouped 5-fold), broken down by segment. **Bias** is the median of predicted ÷ actual − 1: positive means the model over-prices that group.
""")

code(r"""
oof = np.zeros(len(y))
for a, b in FOLDS:
    oof[b] = lgb.LGBMRegressor(**LGB).fit(X.iloc[a], y[a]).predict(X.iloc[b])
E = pd.DataFrame({"actual": np.exp(y), "pred": np.exp(oof), "room": A["room_type"].values,
                  "reviews": np.where(A["number_of_reviews"].values > 0, "has reviews", "no reviews"),
                  "min_stay": np.where(A["minimum_nights"].values >= 28, "monthly (28+)", "short stay"),
                  "price_band": pd.qcut(np.exp(y), 5, labels=["cheapest 20%", "2nd", "middle", "4th", "priciest 20%"]).astype(str),
                  "hood": A["neighbourhood"].values})
E["abs_err"] = (E["pred"] - E["actual"]).abs(); E["ratio"] = E["pred"] / E["actual"]
def seg(col):
    g = E.groupby(col).agg(listings=("actual", "size"), MAE=("abs_err", "mean"), bias_pct=("ratio", lambda r: (np.median(r) - 1) * 100),
                           within_20=("ratio", lambda r: np.mean(np.abs(r - 1) <= 0.2) * 100))
    return g.round(1)
segs = {c: seg(c) for c in ["price_band", "room", "reviews", "min_stay"]}
for c, t in segs.items():
    print("\n", c); print(t.to_string())
""")

code(r"""
hood_err = seg("hood").query("listings >= 40").sort_values("bias_pct")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(E["actual"], E["pred"], s=4, alpha=.3); ax[0].plot([30, 2000], [30, 2000], "k--", lw=1)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_title("Out-of-fold predicted vs actual"); ax[0].set_xlabel("actual $"); ax[0].set_ylabel("predicted $")
pb = segs["price_band"].loc[["cheapest 20%", "2nd", "middle", "4th", "priciest 20%"]]
ax[1].bar(pb.index, pb["bias_pct"], color=np.where(pb["bias_pct"] > 0, "#e76f51", "#3b6fb6")); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("Median bias by actual price band (%)"); plt.tight_layout(); plt.show()
hood_err.head(4).round(1)
""")

md(r"""
**Regression to the middle.** The model over-prices the cheapest listings and under-prices the most expensive. This is the typical failure of any model that is unsure: it pulls predictions toward the average. Part of it is unavoidable. Very cheap and very expensive prices often come from things the data can't see, like a promotion, a luxury renovation, or a listing priced for a special event.
""")

md(r"""
## 8. Verdict and what comes next

The tables above give the exact figures; the points below interpret them.

**Overfitting**
* **The model memorises, but that isn't what limits it.** LightGBM's training R² is far above its held-out R², like every tree model. Yet it still has the best held-out error, and its errors move little across the 18 hyperparameter settings.
* **Selection overfitting is the real risk.** Pure-noise features sometimes pass the keep rule, so small gains (well under a dollar) can't be distinguished from luck. The nested test gives the honest value of the merge; gains that survive it are real.
* **Time drift is the biggest practical risk.** A summer-trained model over-prices September. Training on both quarters reduces the bias but doesn't remove it, and the 80% range should be checked on new data before it is trusted.

**Errors to watch**
* Systematic regression to the middle: cheap listings over-priced, luxury ones under-priced.
* Listings without reviews and monthly rentals are harder.
* Inputs the calculator must ask for honestly: the quoted night's day of week is a real driver, so the tool should ask "which night?" rather than assume.
""")

code(r"""
lgb_lc = slope.loc["LightGBM"]; mlp_lc = slope.loc["Neural network (MLP)"]
bundle = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
bundle["overfit"] = {
    "gap": gapdf.groupby("model")[["train_R2", "heldout_R2", "heldout_MAE"]].mean().round(4).reset_index().to_dict(orient="records"),
    "learning_curve": curve.round(3).to_dict(orient="records"),
    "noise": {"tested": int(len(noise_df)), "kept": int(noise_df["would be kept"].sum()), "max_gain": float(-noise_df["change $"].min()),
              "median_abs_change": float(noise_df["change $"].abs().median())},
    "nested": {"non_nested_change": float(merged_err.mean() - base_err.mean()), "nested_change": float(nested["change $"].mean()),
               "folds_better": int((nested["change $"] < 0).sum()), "chosen": nested["ideas chosen inside the fold"].tolist()},
    "oot": oot.reset_index().round(3).to_dict(orient="records"),
    "sens": {"min": float(sens["CV MAE"].min()), "max": float(sens["CV MAE"].max())},
    "segments": {c: t.reset_index().to_dict(orient="records") for c, t in segs.items()},
    "cv_mae": float(merged_err.mean()),
}
payload = json.dumps(bundle, separators=(",", ":"))
(OUT / "data.js").write_text("window.PRICING = " + payload + ";\n")
print(f"added 'overfit' to interface/data.js ({len(payload) / 1e6:.2f} MB)")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "boston_airbnb_overfitting_check.ipynb"
nbf.write(nb, out); print("wrote", out)
