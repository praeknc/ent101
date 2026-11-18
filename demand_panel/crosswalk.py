"""Model / provider / package crosswalk.

The crosswalk is the merge key between every demand source and Part A. Each
canonical `model_id` lists the identifier it carries in each source
(OpenRouter slug, Hugging Face repo, Part A id) plus free-text aliases.
Unmatched ids are never dropped silently: `resolve` returns them so they can be
added to data/reference/model_crosswalk.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r":(free|beta|extended|thinking|nitro|floor)$", "", s)  # OpenRouter variant suffixes
    s = s.replace("_", "-").replace(" ", "-")
    return s


def variant_flag(slug: str) -> str:
    m = re.search(r":([a-z]+)$", str(slug).strip().lower())
    return m.group(1) if m else ""


class Crosswalk:
    def __init__(self, models: pd.DataFrame, packages: pd.DataFrame | None = None):
        self.models = models.fillna("").astype(str)
        self.packages = packages.fillna("").astype(str) if packages is not None else pd.DataFrame()
        self._lookup: dict[str, str] = {}
        for _, r in self.models.iterrows():
            keys = [r["model_id"], r["openrouter_slug"], r["hf_repo_id"], r["part_a_model_id"]]
            keys += [a for a in r["aliases"].split("|") if a]
            # bare name without provider prefix, e.g. "gpt-4o"
            keys += [k.split("/", 1)[1] for k in keys if "/" in k]
            for k in keys:
                if not k:
                    continue
                nk = _norm(k)
                prev = self._lookup.get(nk)
                if prev and prev != r["model_id"]:
                    raise ValueError(f"Alias '{k}' maps to both {prev} and {r['model_id']}")
                self._lookup[nk] = r["model_id"]

    @classmethod
    def load(cls, ref_dir: Path | str = REF_DIR) -> "Crosswalk":
        ref_dir = Path(ref_dir)
        models = pd.read_csv(ref_dir / "model_crosswalk.csv", dtype=str)
        pk = ref_dir / "package_map.csv"
        packages = pd.read_csv(pk, dtype=str) if pk.exists() else None
        return cls(models, packages)

    # ------------------------------------------------------------------ models
    def model_id(self, raw: str) -> str:
        return self._lookup.get(_norm(raw), "")

    def provider_of(self, model_id: str) -> str:
        hit = self.models.loc[self.models["model_id"] == model_id, "provider_id"]
        return hit.iloc[0] if len(hit) else ""

    def attr(self, model_id: str, col: str) -> str:
        hit = self.models.loc[self.models["model_id"] == model_id, col]
        return hit.iloc[0] if len(hit) else ""

    def resolve(self, raw_ids) -> tuple[pd.Series, list[str]]:
        """Map raw ids to canonical model_ids. Returns (mapped, sorted list of unmatched raw ids)."""
        raw = pd.Series(raw_ids, dtype=str)
        mapped = raw.map(self.model_id).fillna("")
        unmatched = sorted(set(raw[mapped == ""]))
        return mapped, unmatched

    def part_a_id(self, model_id: str) -> str:
        return self.attr(model_id, "part_a_model_id") or model_id

    def hf_repos(self) -> pd.DataFrame:
        return self.models.loc[self.models["hf_repo_id"] != "", ["model_id", "provider_id", "hf_repo_id"]]

    # ------------------------------------------------------------------ packages
    def packages_for(self, ecosystem: str) -> pd.DataFrame:
        if self.packages.empty:
            return self.packages
        return self.packages.loc[self.packages["ecosystem"] == ecosystem]

    def provider_of_package(self, ecosystem: str, package: str) -> str:
        p = self.packages_for(ecosystem)
        hit = p.loc[p["package"] == package, "provider_id"]
        return hit.iloc[0] if len(hit) else ""
