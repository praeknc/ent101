"""Builds notebooks/boston_airbnb_phase3_text_market.ipynb (Phase 3, Steps 2b-5).

Run:  python scripts/download_data.py && python scripts/download_photos.py
      python scripts/build_text_notebook.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/boston_airbnb_phase3_text_market.ipynb
Steps 4b and 5 use pretrained models from Hugging Face (via fastembed). If they can't be downloaded,
those rungs are skipped and the notebook says so.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 3, Steps 2b-5: market data, text and photos

Steps 1-2 (previous notebook) showed that describing the listing itself cut the error by about a quarter, while extra location signals added little. This notebook tests the remaining data sources, one rung at a time, on top of the recommended Steps 1-2 feature set:

| Rung | Adds | Source | Method |
|---|---|---|---|
| 2b | Neighbourhood home value and its 12-month change | Zillow ZHVI | joined by neighbourhood and month |
| 3 | Seasonality | `calendar.csv.gz` | **not possible**: see section 1 |
| 4a | Description and review text | listing descriptions; 30 most recent reviews before each snapshot | TF-IDF → SVD, plus an interpretable aspect lexicon |
| 4b | Same text, read by a pretrained language model | as above | `BAAI/bge-small-en-v1.5` sentence embeddings |
| 5 | Cover photo | listing photo (320 px) | `CLIP ViT-B/32` image embeddings |

Every rung is judged by **host-grouped 5-fold cross-validation with LightGBM**, the best model from Steps 1-2. All text and image features are fitted **inside each fold on training rows only**.
""")

code(r"""
import json, os, sys, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
DATA, OUT, CACHE, SEED = Path("../data"), Path("../interface"), Path("../data/raw/cache"), 42
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "../src"); sys.path.insert(0, "../scripts")
import pricing_features as pf
import enriched_features as ef
from download_photos import photo_path

official = pd.read_csv(DATA / "neighbourhoods.csv")["neighbourhood"]
geo = pf.load_geo(DATA / "neighbourhoods.geojson")
H = pf.model_neighbourhoods(official)
_js = (OUT / "data.js").read_text(); BUNDLE = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
SNAP = {"2026Q2": ("listings_detailed_2026Q2_jun.csv.gz", "2026-06-15"), "2026Q3": ("listings_detailed_2026Q3_sep.csv.gz", "2026-09-19")}
""")

md(r"""
## 1. What the new downloads contain

Before using any file, check that it holds what the plan assumes.
""")

code(r"""
cal_check = []
for q in ["2025Q4", "2026Q1", "2026Q2", "2026Q3"]:
    c = pd.read_csv(DATA / "raw" / q / "calendar.csv.gz", nrows=200000)
    has_price = "price" in c.columns
    cal_check.append({"snapshot": q, "columns": ", ".join(c.columns), "price column": has_price,
                      "price filled": f"{c['price'].notna().mean():.0%}" if has_price else "n/a"})
pd.DataFrame(cal_check)
""")

md(r"""
**Step 3 is not possible with this data.** Inside Airbnb no longer publishes nightly prices in `calendar.csv`: the December file has an empty price column, and June and September have none at all. What remains is whether each night is available, which mostly reflects *bookings*, an outcome of the price, so it can't be an input either. Seasonality therefore stays limited to the June vs September contrast learned in Phase 2.
""")

md("## 2. Rebuild the Steps 1-2 feature set")

code(r"""
frames, supply = [], []
for q, (f, _) in SNAP.items():
    raw = ef.load_detailed(DATA / "detailed" / f)
    a, _ = pf.precheck(raw, official, geo, require_price=False); a["quarter"] = q; supply.append(a)
    d, _ = pf.precheck(raw, official, geo); d["quarter"] = q; frames.append(d)
A = pd.concat(frames, ignore_index=True); SUPPLY = pd.concat(supply, ignore_index=True)
PROPS = sorted(set(ef.PROPERTY_GROUPS.values())) + ["other"]
BASE = pd.concat([pf.build_features(A, H),
                  ef.time_zone_features(A, DATA / "quarterly", BUNDLE["quarterly"]["zones"]["centers_km"]),
                  ef.detailed_features(A, PROPS), ef.landmark_features(A), ef.density_features(A, SUPPLY)], axis=1)
y, groups = np.log(A["price"].values), A["host_id"].values
print(A.shape[0], "priced rows,", BASE.shape[1], "Steps 1-2 features (neighbours' market price left out, as recommended)")
""")

md(r"""
## 3. Rung 2b: housing market (Zillow ZHVI)

The home value index of the listing's neighbourhood at the **last month-end before the snapshot** (May for June, August for September), plus its 12-month change. Zillow splits Dorchester in two and has no Longwood area, so those are averaged from the matching Zillow areas (`ZILLOW_MAP`). Zillow's rent index (ZORI) is published by ZIP code only, and the listings carry no ZIP, so it is not used.
""")

code(r"""
zhvi = pd.read_csv(DATA / "market" / "Neighborhood_zhvi.csv")
Z = ef.zhvi_features(A, zhvi)
print("coverage:", Z.notna().mean().round(3).to_dict())
zt = A.assign(**Z).groupby("neighbourhood")[["zhvi_log", "zhvi_yoy_pct"]].mean()
zt["home_value_$k"] = (np.exp(zt["zhvi_log"]) / 1000).round(0)
print("Spearman with log price:", round(Z["zhvi_log"].corr(pd.Series(y), method="spearman"), 3))
zt.sort_values("home_value_$k", ascending=False).round(1).head(8)
""")

md(r"""
## 4. Rung 4a: text, classic NLP

* **Reviews:** the 30 most recent reviews **posted before each snapshot date**, taken from that snapshot's `reviews.csv.gz`. No review written after the price was observed is used.
* **Price words removed.** Words like *price, value, worth, expensive, cheap* are dropped from the text. Guests' comments on value are a reaction to the price, so reading them would leak the answer.
* **Aspect lexicon:** the share of a listing's reviews that mention cleanliness, dirt, noise, quiet, space, smallness, location, transit, views, luxury, the host or problems. Easy to explain to a host.
* **TF-IDF → SVD:** 16 components for the description, 24 for reviews, fitted on each fold's training rows.
""")

code(r"""
t0 = time.time()
REVTXT = pd.Series("", index=A.index); ASP = pd.DataFrame(index=A.index)
for q, (_, date) in SNAP.items():
    rv = pd.read_csv(DATA / "raw" / q / "reviews.csv.gz")
    t, asp = ef.review_text_by_listing(rv, date)
    m = (A["quarter"] == q).values
    REVTXT[m] = A.loc[m, "id"].map(t).fillna("").values
    ASP = ASP.combine_first(pd.DataFrame(asp.reindex(A.loc[m, "id"]).values, index=A.index[m], columns=asp.columns))
DESC = A["description"].fillna("").str.replace(r"<br\s*/?>", " ", regex=True)
print(f"{time.time() - t0:.0f}s | listings with review text: {(REVTXT.str.len() > 0).mean():.1%}")
STOP = list(ENGLISH_STOP_WORDS | set(ef.PRICE_WORDS))
asp_corr = ASP.drop(columns=["n_reviews_text", "review_len_mean"]).corrwith(pd.Series(y, index=A.index), method="spearman").sort_values()
asp_corr.round(3).to_frame("Spearman with log price")
""")

code(r"""
def tfidf_svd(text, a, b, k, prefix):
    v = TfidfVectorizer(min_df=10, max_features=30000, ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP)
    Xa, Xb = v.fit_transform(text.iloc[a]), v.transform(text.iloc[b])
    s = TruncatedSVD(k, random_state=SEED).fit(Xa)
    out = np.zeros((len(text), k)); out[a], out[b] = s.transform(Xa), s.transform(Xb)
    return pd.DataFrame(out, index=text.index, columns=[f"{prefix}{i}" for i in range(k)])

def pca_block(E, a, b, k, prefix, mask=None):
    # PCA of pretrained embeddings fitted on training rows; rows without an embedding stay NaN
    mask = np.ones(len(E), bool) if mask is None else mask
    fa = a[mask[a]]
    p = PCA(k, random_state=SEED).fit(E[fa])
    out = np.full((len(E), k), np.nan)
    for idx in (a, b):
        ok = idx[mask[idx]]
        out[ok] = p.transform(E[ok])
    return pd.DataFrame(out, index=A.index, columns=[f"{prefix}{i}" for i in range(k)])
""")

md(r"""
## 5. Rungs 4b and 5: pretrained deep-learning models

* **Text:** `BAAI/bge-small-en-v1.5`, a 33M-parameter sentence-embedding model. It reads the description and the recent reviews (truncated to the model's 512 tokens) and returns a 384-number summary of their meaning.
* **Photos:** `CLIP ViT-B/32` (OpenAI's image-text model, ONNX build) turns each cover photo into 512 numbers that capture style, light, staging and room type.

Both run through `fastembed` (ONNX, no PyTorch needed) and are frozen: we don't fine-tune them on ~7,000 prices, which would mostly memorise. Embeddings are cached in `data/raw/cache/`. If a model can't be downloaded, its rung is skipped.
""")

code(r"""
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
def cached(name, fn):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    try:
        E = fn(); np.save(path, E); return E
    except Exception as e:  # noqa: BLE001 - a blocked download should skip the rung, not stop the notebook
        print(f"{name}: skipped ({type(e).__name__}: {str(e)[:120]})"); return None

def embed_text(texts):
    from fastembed import TextEmbedding
    m = TextEmbedding("BAAI/bge-small-en-v1.5")
    return np.array(list(m.embed([t[:2000] if t else "no text" for t in texts], batch_size=64)))

t0 = time.time()
E_DESC = cached("bge_desc", lambda: embed_text(list(DESC)))
E_REV = cached("bge_reviews", lambda: embed_text(list(REVTXT)))
has_rev = (REVTXT.str.len() > 0).values

photo_files = [photo_path(u) for u in A["picture_url"]]
has_photo = np.array([p.exists() for p in photo_files])
def embed_photos():
    from fastembed import ImageEmbedding
    m = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
    E = np.full((len(A), 512), np.nan)
    idx = np.where(has_photo)[0]
    E[idx] = np.array(list(m.embed([str(photo_files[i]) for i in idx], batch_size=32)))
    return E
E_IMG = cached("clip_photos", embed_photos)
print(f"{time.time() - t0:.0f}s | photos on disk for {has_photo.mean():.1%} of rows |",
      "text model:", "ok" if E_DESC is not None else "unavailable", "| image model:", "ok" if E_IMG is not None else "unavailable")
""")

md(r"""
## 6. The ladder, with host-grouped 5-fold CV

Each rung adds to the one before. The same folds are used for every rung, so the per-fold difference shows whether a gain is consistent or noise.
""")

code(r"""
LGB = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED, verbose=-1)
rungs, rows, imps = [], [], []
for k, (a, b) in enumerate(GroupKFold(5).split(A, y, groups)):
    blocks = {"Steps 1-2 base": BASE, "+ 2b Zillow home values": Z,
              "+ 4a description (TF-IDF)": tfidf_svd(DESC, a, b, 16, "desc_tf_"),
              "+ 4a review aspects": ASP,
              "+ 4a review text (TF-IDF)": tfidf_svd(REVTXT, a, b, 24, "rev_tf_")}
    if E_DESC is not None:
        blocks["+ 4b description (bge-small)"] = pca_block(E_DESC, a, b, 16, "desc_bge_")
        blocks["+ 4b reviews (bge-small)"] = pca_block(E_REV, a, b, 16, "rev_bge_", mask=has_rev)
    if E_IMG is not None:
        blocks["+ 5 photo (CLIP)"] = pca_block(E_IMG, a, b, 16, "img_clip_", mask=has_photo)
    X = None
    for name, blk in blocks.items():
        X = blk if X is None else pd.concat([X, blk], axis=1)
        m = lgb.LGBMRegressor(**LGB).fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
        rows.append({"fold": k, "rung": name, "MAE_$": mean_absolute_error(np.exp(y[b]), np.exp(p)),
                     "RMSE_log": float(np.sqrt(np.mean((p - y[b]) ** 2))), "n_features": X.shape[1]})
    if k == 0: rungs = list(blocks)
    imps.append(pd.Series(m.booster_.feature_importance("gain"), X.columns))
cv = pd.DataFrame(rows)
piv = cv.pivot(index="fold", columns="rung", values="MAE_$")[rungs]
ladder = pd.DataFrame({"CV MAE ($/night)": piv.mean(), "± std": piv.std(),
                       "change vs previous": piv.diff(axis=1).mean(), "folds improved": (piv.diff(axis=1) < 0).sum(),
                       "features": cv.groupby("rung")["n_features"].first()[rungs],
                       "RMSE (log)": cv.groupby("rung")["RMSE_log"].mean()[rungs]})
ladder.iloc[0, 2:4] = np.nan
ladder.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(rungs)), piv.mean().values, marker="o", color="#3b6fb6")
for i, v in enumerate(piv.mean().values): ax.text(i, v + 0.15, f"${v:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(rungs)), [r.replace("+ ", "+\n") for r in rungs], fontsize=8); ax.set_title("Host-grouped CV MAE by rung (lower is better)")
plt.tight_layout(); plt.show()
""")

code(r"""
gain = pd.concat(imps, axis=1).mean(axis=1)
def family(c):
    for pre, fam in [("desc_tf_", "Description text"), ("desc_bge_", "Description text"), ("rev_tf_", "Review text"),
                     ("rev_bge_", "Review text"), ("asp_", "Review aspects"), ("img_clip_", "Photo"), ("zhvi_", "Housing market")]:
        if c.startswith(pre): return fam
    return "Steps 1-2 features"
fam_gain = (gain.groupby(gain.index.map(family)).sum() / gain.sum() * 100).sort_values()
fam_gain.round(1).to_frame("% of LightGBM split gain (full model)")
""")

md(r"""
## 7. Verdict for Steps 2b-5

The ladder table above gives the exact numbers for this run; the bullets below summarise them.

* **Housing market (Zillow) adds nothing within Boston.** Home values vary by neighbourhood, and the model already knows the neighbourhood. Its value is for **comparing cities**, where it explains why the same flat costs more in one market than another.
* **Text helps a little.** The description and the review wording each lower the error by under a dollar a night, and together by about 2%. Results vary more from fold to fold once review text is in, so treat the gain as small but real.
* **Review aspect scores add almost no accuracy** once the wording is in, because the same signal is already in the words. They stay useful for *explaining* a price: listings whose reviews mention location, luxury and space are priced higher; those whose reviews mention transit and quiet are priced lower (they tend to be outside the centre).
* **Seasonality could not be added:** the calendar files no longer contain prices.
* **Pretrained models (rungs 4b and 5)** appear in the ladder only when the models could be downloaded. In the current run they were skipped: the network policy blocks Hugging Face's file CDN (`us.aws.cdn.hf.co`). Allowing it and rerunning adds both rungs; the photos are already downloaded.
""")

code(r"""
bundle = json.loads(_js[_js.index("{"): _js.rindex("}") + 1])
bundle["phase3b"] = {
    "rungs": rungs,
    "ladder": ladder.reset_index().rename(columns={"index": "rung", "rung": "rung"}).replace({np.nan: None}).round(4).to_dict(orient="records"),
    "family_gain_pct": fam_gain.sort_values(ascending=False).round(2).to_dict(),
    "aspect_corr": asp_corr.round(4).to_dict(),
    "calendar": cal_check,
    "models_available": {"text": E_DESC is not None, "image": E_IMG is not None},
    "photo_coverage": float(has_photo.mean()),
}
payload = json.dumps(bundle, separators=(",", ":"))
(OUT / "data.js").write_text("window.PRICING = " + payload + ";\n")
print(f"added 'phase3b' to interface/data.js ({len(payload) / 1e6:.2f} MB)")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "boston_airbnb_phase3_text_market.ipynb"
nbf.write(nb, out); print("wrote", out)
