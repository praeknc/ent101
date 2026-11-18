"""Minimal Markdown table writer (avoids a tabulate dependency)."""
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    esc = lambda v: "" if pd.isna(v) else str(v).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(esc(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)
