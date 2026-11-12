# Frontier LLM API pricing history

`frontier_pricing.py` covers API prices from GPT-3 (2020) to today. The labs are OpenAI, Anthropic, Google, xAI, Meta, Amazon, Mistral and Cohere. The Chinese labs are DeepSeek, Alibaba/Qwen, ByteDance, Baidu, Moonshot (Kimi), Zhipu (GLM) and MiniMax. Open-weight models are flagged.

Standard library only, so no install step.

## Usage

```bash
python3 pricing/frontier_pricing.py summary                 # one row per lab
python3 pricing/frontier_pricing.py table --country CN      # Chinese labs
python3 pricing/frontier_pricing.py table --open            # open-weight models
python3 pricing/frontier_pricing.py timeline --lab anthropic
python3 pricing/frontier_pricing.py yearly                  # min/median/max per year, US vs CN
python3 pricing/frontier_pricing.py cheapest --top 15       # or --expensive
python3 pricing/frontier_pricing.py chart --out prices.svg  # log-scale chart; hover for details
python3 pricing/frontier_pricing.py export --format json --out prices.json
python3 pricing/frontier_pricing.py live --save             # current prices from OpenRouter
```

Filters work on every command: `--lab`, `--country`, `--open` or `--closed`, `--since 2024`, `--min-confidence high` and `--launches-only`.

## Data notes

- Each row in `data/llm_api_prices.csv` is one price event: a launch, a price cut or a price change. Prices are USD per 1M tokens. **Blended** = (3 × input + output) / 4.
- Prices are list prices for prompts at standard length. They leave out cache, batch and long-context tiers. CNY prices are converted at about 7.2 CNY per USD.
- Open-weight models without a first-party API use a typical third-party host price, and their rows say so in `notes`.
- The `confidence` column means:
  - `high`: the official launch price.
  - `medium`: a converted price, a typical host price, or a date that is not fully certain.
  - `low`: an approximate figure.
- Rows with no date are Anthropic's 2026 models. Their prices come from Anthropic's docs, but their release dates still need to be filled in.
- `live --save` writes a dated snapshot to `data/snapshots/` and shows what changed since the last one. If you run it on a schedule, your price history keeps growing.
