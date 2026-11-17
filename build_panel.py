#!/usr/bin/env python3
"""
build_panel.py  -  merge the price sources into one colour-coded Excel workbook

INPUTS (./pricing_data/)
  prices_history.csv   Wayback snapshots of official pricing pages  (source W)
  prices_litellm.csv   LiteLLM git-history price events             (source L)
  prices.csv           live official pages, optional: --live PATH   (source Live)
  fx_cny_usd.csv       monthly CNY per USD (FRED / Federal Reserve H.10)

OUTPUT
  pricing_data/AI_API_price_panel.xlsx
    README              legend, colours, column meanings, caveats
    Panel (original)    every price row in the currency the source published
    Panel (USD)         same rows; CNY converted with formulas via 'FX rates'
    FX rates            monthly CNY per USD used for the conversion
    Compare W vs L      Wayback price vs the LiteLLM price in force that day

USAGE
  python3 build_panel.py
  python3 build_panel.py --live pricing_data/prices.csv
"""
import argparse
import csv
import re
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

D = Path("pricing_data")
FONT = "Arial"
SRC = {  # code: (label, fill colour, description)
    "W": ("Wayback", "DDEBF7", "Archived official pricing page (Internet Archive snapshot)"),
    "L": ("LiteLLM", "FCE4D6", "LiteLLM community price file, git history (secondary source)"),
    "Live": ("Live", "E2EFDA", "Official pricing page fetched live by scrape_ai_pricing.py"),
}
P1 = "Panel (original)"
P2 = "Panel (USD)"
FXS = "FX rates"


# ------------------------------------------------------------------ helpers
def model_key(name):
    """Normalise names so the same model matches across sources:
    'Claude Opus 4.5' / 'claude-opus-4-5-20251101' -> 'claude-opus-4-5',
    'Gemini 2.5 Pro' / 'gemini/gemini-2.5-pro' -> 'gemini-2-5-pro'."""
    s = name.lower()
    s = re.sub(r'\(.*?\)|\[.*?\]', ' ', s)
    s = re.sub(r'\s+-\s+deprecated|\bdeprecated\b', ' ', s)
    s = s.split("/")[-1]
    s = re.sub(r'[\s._]+', '-', s.strip())
    s = re.sub(r'-(\d{8}|\d{4}-\d{2}-\d{2})$', '', s)
    s = re.sub(r'^claude-(\d(?:-\d)?)-(opus|sonnet|haiku)', r'claude-\2-\1', s)
    return s.strip("-")


def fnum(x):
    try:
        return float(x) if str(x).strip() != "" else None
    except ValueError:
        return None


def tidy_flags(reason):
    drop = {"non-authoritative source", "secondary source (LiteLLM community price file)"}
    return "; ".join(x for x in reason.split("; ") if x and x not in drop)


def unit_note(u):
    return {"1K": "per 1K on page; x1000 to per 1M", "1M": "per 1M",
            "1M chars": "per 1M CHARACTERS (not tokens)"}.get(u, u)


def load(path, code):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = date.fromisoformat(r["retrieved_at"][:10])
        rows.append(dict(
            src=code, provider=r["provider"], model=r["model"], key=model_key(r["model"]),
            date=d, month=d.strftime("%Y-%m"), section=r.get("section", ""),
            event=r.get("event") or ("snapshot" if code == "W" else "live"),
            inp=fnum(r["input_per_1m"]), out=fnum(r["output_per_1m"]),
            cache=fnum(r.get("cache_read_per_1m", "")),
            cur=r["currency"], unit=r["unit_detected"],
            flags=tidy_flags(r["review_reason"]), url=r["source_url"]))
    return rows


def label_unlabelled_tiers(rows):
    """Archived 2026 OpenAI pages list Standard / Batch / Flex / Priority / Fast /
    fine-tuning prices as separate tables with no tier heading the parser can see,
    so those rows arrive with an empty section. Table order is not a safe guide
    (older models appear only in later tables), so they are labelled, flagged and
    kept out of the W-vs-L comparison until the parser can read the tier."""
    for r in rows:
        if r["src"] == "W" and r["provider"] == "openai" and not r["section"]:
            r["section"] = "tier not labelled on archived page"
            r["flags"] = "; ".join(x for x in (r["flags"], "tier unknown: Standard, Batch, "
                                   "Flex, Priority or fine-tuned (archived page)") if x)
    return rows


def dedupe(rows):
    """The same page archived under two URLs on one day gives identical rows."""
    seen, out = set(), []
    for r in rows:
        k = (r["src"], r["provider"], r["date"], r["model"], r["section"], r["inp"], r["out"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ------------------------------------------------------------------ styling
THIN = Side(style="thin", color="BFBFBF")
HEAD_FILL = PatternFill("solid", fgColor="1F3864")


def header(ws, cols, widths):
    for j, (c, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"


def style_row(ws, i, ncols, src):
    fill = PatternFill("solid", fgColor=SRC[src][1])
    for j in range(1, ncols + 1):
        c = ws.cell(row=i, column=j)
        c.fill = fill
        c.font = Font(name=FONT, size=10, bold=(j == 1))
        c.border = Border(bottom=THIN)


PRICE_FMT = '#,##0.00####;-#,##0.00####;0'
FX_FMT = '0.0000'


# ------------------------------------------------------------------ sheets
def sheet_original(wb, rows):
    ws = wb.create_sheet(P1)
    cols = ["Source", "Provider", "Model (as published)", "Model key", "Date", "Month",
            "Section / tier", "Event", "Input / 1M tokens", "Output / 1M tokens",
            "Cached input / 1M", "Currency", "Unit on page", "Flags", "Source URL"]
    header(ws, cols, [8, 10, 34, 26, 11, 9, 26, 12, 13, 13, 12, 9, 22, 48, 40])
    for i, r in enumerate(rows, 2):
        vals = [r["src"], r["provider"], r["model"], r["key"], r["date"], r["month"],
                r["section"], r["event"], r["inp"], r["out"], r["cache"], r["cur"],
                unit_note(r["unit"]), r["flags"], r["url"]]
        for j, v in enumerate(vals, 1):
            ws.cell(row=i, column=j, value=v)
        style_row(ws, i, len(cols), r["src"])
        ws.cell(row=i, column=5).number_format = "yyyy-mm-dd"
        for j in (9, 10, 11):
            ws.cell(row=i, column=j).number_format = PRICE_FMT
    ws.cell(row=1, column=9).comment = Comment(
        "Price per 1M tokens in the column 'Currency'. Pages quoting per 1K tokens were "
        "multiplied by 1000 (see 'Unit on page').", "build_panel")
    return ws


def sheet_usd(wb, rows, fx_last_row):
    ws = wb.create_sheet(P2)
    cols = ["Source", "Provider", "Model (as published)", "Model key", "Date", "Month",
            "Section / tier", "Event", "FX used (CNY per USD)", "Input USD / 1M",
            "Output USD / 1M", "Cached input USD / 1M", "Original currency", "Unit on page",
            "Flags"]
    header(ws, cols, [8, 10, 34, 26, 11, 9, 26, 12, 12, 13, 13, 12, 10, 22, 48])
    fxa, fxb = f"'{FXS}'!$A$2:$A${fx_last_row}", f"'{FXS}'!$B$2:$B${fx_last_row}"
    o = f"'{P1}'!"
    for i, r in enumerate(rows, 2):
        for j, v in enumerate([r["src"], r["provider"], r["model"], r["key"], r["date"],
                               r["month"], r["section"], r["event"]], 1):
            ws.cell(row=i, column=j, value=v)
        # FX: 1 for USD rows; that month's CNY-per-USD rate for CNY rows
        ws.cell(row=i, column=9,
                value=f'=IF(M{i}="CNY",IFERROR(INDEX({fxb},MATCH(F{i},{fxa},0)),""),1)')
        for j, src_col in ((10, "I"), (11, "J"), (12, "K")):
            ws.cell(row=i, column=j,
                    value=f'=IF(OR({o}{src_col}{i}="",$I{i}=""),"",{o}{src_col}{i}/$I{i})')
            ws.cell(row=i, column=j).number_format = PRICE_FMT
        ws.cell(row=i, column=13, value=r["cur"])
        ws.cell(row=i, column=14, value=unit_note(r["unit"]))
        ws.cell(row=i, column=15, value=r["flags"])
        style_row(ws, i, len(cols), r["src"])
        ws.cell(row=i, column=5).number_format = "yyyy-mm-dd"
        ws.cell(row=i, column=9).number_format = FX_FMT
    ws.cell(row=1, column=9).comment = Comment(
        "Monthly average CNY per USD for the row's month, looked up in 'FX rates' "
        "(FRED series EXCHUS, Federal Reserve H.10). 1 for rows already in USD.", "build_panel")
    return ws


def sheet_fx(wb, fx):
    ws = wb.create_sheet(FXS)
    header(ws, ["Month", "CNY per USD (monthly avg)", "Source"], [10, 16, 70])
    src = ("FRED series EXCHUS (Board of Governors of the Federal Reserve System, H.10), "
           "monthly average, via github.com/datasets/exchange-rates data/monthly.csv")
    for i, (m, v) in enumerate(fx, 2):
        ws.cell(row=i, column=1, value=m).font = Font(name=FONT, size=10)
        c = ws.cell(row=i, column=2, value=v)
        c.font = Font(name=FONT, size=10, color="0000FF")  # hardcoded input
        c.number_format = FX_FMT
        ws.cell(row=i, column=3, value=src if i == 2 else "").font = Font(name=FONT, size=9)
    return ws, len(fx) + 1


def price_in_force(lrows, d):
    """LiteLLM price for a key on date d: last first_seen/changed/cache_changed on or
    before d, unless the model was removed after that and before d."""
    cur = None
    for r in lrows:
        if r["date"] > d:
            break
        cur = None if r["event"] == "removed" else r
    return cur


def sheet_compare(wb, rows):
    ws = wb.create_sheet("Compare W vs L")
    cols = ["Provider", "Model key", "Wayback date", "Wayback model", "Wayback input",
            "Wayback output", "LiteLLM model", "LiteLLM recorded", "LiteLLM input",
            "LiteLLM output", "Result"]
    header(ws, cols, [10, 24, 12, 30, 12, 12, 30, 13, 12, 12, 10])
    skip = re.compile(r'batch|flex|fast|priority|fine-?tun|cache|ultrafast|not labelled', re.I)
    not_base = re.compile(r'prompts? (over|longer|above)|starting|through|until', re.I)
    by_key = {}
    for r in rows:
        if r["src"] == "L":
            by_key.setdefault((r["provider"], r["key"]), []).append(r)
    for v in by_key.values():
        v.sort(key=lambda r: r["date"])
    i = 2
    for r in rows:
        if r["src"] != "W" or r["cur"] != "USD" or r["unit"] != "1M" or "[" in r["model"] \
                or skip.search(r["section"]) or not_base.search(r["model"]) \
                or r["inp"] is None or r["out"] is None:
            continue
        lst = by_key.get((r["provider"], r["key"]))
        if not lst:
            continue
        lr = price_in_force(lst, r["date"])
        if not lr or lr["inp"] is None or lr["out"] is None:
            continue
        vals = [r["provider"], r["key"], r["date"], r["model"], r["inp"], r["out"],
                lr["model"], lr["date"], lr["inp"], lr["out"]]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = Font(name=FONT, size=10)
        ws.cell(row=i, column=11,
                value=f'=IF(AND(ABS(E{i}-I{i})<0.000001,ABS(F{i}-J{i})<0.000001),"match","DIFFERS")')
        ws.cell(row=i, column=11).font = Font(name=FONT, size=10, bold=True)
        for j in (3, 8):
            ws.cell(row=i, column=j).number_format = "yyyy-mm-dd"
        for j in (5, 6, 9, 10):
            ws.cell(row=i, column=j).number_format = PRICE_FMT
        for j in (4, 5, 6):
            ws.cell(row=i, column=j).fill = PatternFill("solid", fgColor=SRC["W"][1])
        for j in (7, 8, 9, 10):
            ws.cell(row=i, column=j).fill = PatternFill("solid", fgColor=SRC["L"][1])
        i += 1
    from openpyxl.formatting.rule import FormulaRule
    if i > 2:
        ws.conditional_formatting.add(
            f"K2:K{i - 1}", FormulaRule(formula=['K2="DIFFERS"'],
                                        font=Font(name=FONT, bold=True, color="C00000"),
                                        fill=PatternFill("solid", fgColor="FFC7CE")))
    return ws, i - 2


def sheet_readme(wb, counts, ncompare, has_live):
    ws = wb.active
    ws.title = "README"
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 100
    lines = [
        ("AI API token price panel", None),
        ("Built", date.today().isoformat()),
        ("", None),
        ("SOURCES (row colour + 'Source' column)", None),
    ]
    for code, (label, colour, desc) in SRC.items():
        n = counts.get(code, 0)
        lines.append((f"{code}  =  {label}", f"{desc}. {n} rows" +
                      ("" if code != "Live" or has_live else " (not included yet: run with --live)"),
                      colour))
    lines += [
        ("", None),
        ("SHEETS", None),
        (P1, "Every price row exactly as the source published it (RMB rows stay in CNY)."),
        (P2, "Same rows in the same order; CNY prices converted to USD with formulas "
             "(price / that month's CNY per USD from 'FX rates'). USD rows unchanged."),
        (FXS, "Monthly average CNY per USD, FRED series EXCHUS (Federal Reserve H.10)."),
        ("Compare W vs L", f"{ncompare} Wayback rows (USD, standard tier) next to the LiteLLM "
                           "price in force on the snapshot date. 'DIFFERS' in red = check."),
        ("", None),
        ("COLUMNS", None),
        ("Date", "W: snapshot date of the archived page. L: date LiteLLM changed its file "
                 "(can lag the real price change by days to months). Live: fetch date."),
        ("Event", "W: snapshot. L: first_seen / changed / cache_changed / removed."),
        ("Model key", "Normalised name used to match the same model across sources."),
        ("Input / Output", "Price per 1M tokens. Pages quoting per 1K were x1000 "
                           "(see 'Unit on page'). Base / lowest tier; tiers are in Flags."),
        ("Flags", "Parser notes: tiered pricing, promo / scheduled prices, one price for "
                  "input and output, unusual output/input ratio (possible page typo), etc."),
        ("", None),
        ("CAVEATS", None),
        ("Prefer W over L", "Where they disagree the archived official page is the primary "
                            "source; LiteLLM missed e.g. the 2024 Gemini 1.5 price cuts."),
        ("Snapshots", "W rows for OpenAI / Anthropic / Google are roughly quarterly; a price "
                      "change happened somewhere between two snapshots."),
        ("Markets", "Alibaba (Aliyun China) and Baidu 2024-25 are Chinese-market RMB prices; "
                    "Baidu's 2026 page and LiteLLM's dashscope rows are international USD."),
        ("Characters", "Gemini 1.0 Pro (Mar 2024) is priced per 1M CHARACTERS, not tokens."),
        ("OpenAI 2026 tiers", "Archived 2026 OpenAI pages show Standard / Batch / Flex / "
                              "Priority / fine-tuned prices in unlabelled tables, so those W "
                              "rows say 'tier not labelled'. One model can have several rows "
                              "per date; they are excluded from 'Compare W vs L'."),
    ]
    for i, line in enumerate(lines, 1):
        a = ws.cell(row=i, column=1, value=line[0])
        b = ws.cell(row=i, column=2, value=line[1])
        a.font = Font(name=FONT, size=10, bold=(line[1] is None or len(line) == 3))
        b.font = Font(name=FONT, size=10)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        if len(line) == 3:
            a.fill = PatternFill("solid", fgColor=line[2])
    ws["A1"].font = Font(name=FONT, size=14, bold=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wayback", default=str(D / "prices_history.csv"))
    ap.add_argument("--litellm", default=str(D / "prices_litellm.csv"))
    ap.add_argument("--live", help="live prices.csv to include (source Live)")
    ap.add_argument("--fx", default=str(D / "fx_cny_usd.csv"))
    ap.add_argument("--out", default=str(D / "AI_API_price_panel.xlsx"))
    args = ap.parse_args()

    rows = label_unlabelled_tiers(load(args.wayback, "W")) + load(args.litellm, "L")
    if args.live:
        rows += load(args.live, "Live")
    n0 = len(rows)
    rows = dedupe(rows)
    order = {"W": 0, "Live": 1, "L": 2}
    rows.sort(key=lambda r: (r["provider"], r["key"], r["date"], order[r["src"]], r["section"]))
    fx = [(r["month"], float(r["cny_per_usd"])) for r in csv.DictReader(open(args.fx))]
    missing = sorted({r["month"] for r in rows if r["cur"] == "CNY"} - {m for m, _ in fx})
    if missing:
        print("WARNING: no FX rate for CNY months:", missing)

    wb = Workbook()
    counts = {}
    for r in rows:
        counts[r["src"]] = counts.get(r["src"], 0) + 1
    sheet_original(wb, rows)
    _, fx_last = sheet_fx(wb, fx)
    sheet_usd(wb, rows, fx_last)
    _, ncompare = sheet_compare(wb, rows)
    sheet_readme(wb, counts, ncompare, bool(args.live))
    wb.move_sheet(FXS, offset=2)  # README, originals, USD, FX, compare
    wb.save(args.out)
    print(f"{len(rows)} rows ({n0 - len(rows)} duplicate rows dropped) -> {args.out}")
    print("by source:", counts, f"| compare rows: {ncompare}")


if __name__ == "__main__":
    main()
