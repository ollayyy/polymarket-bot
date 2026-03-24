# Polymarket Trading Bot

A modular Python trading bot for Polymarket prediction markets using limit orders only.

## Structure

```
polymarket-bot/
├── strategies/         # Trading strategies
│   ├── base.py         # BaseStrategy, Signal, SignalType
│   ├── macd.py         # MACD crossover
│   ├── rsi.py          # RSI mean-reversion
│   └── cvd.py          # Cumulative Volume Delta divergence
├── backtesting/
│   └── engine.py       # Event-driven limit-order backtest engine
├── bot/
│   ├── trader.py       # Live trader (CLOB client integration)
│   └── risk_manager.py # Pre-order risk gating
└── deploy/
    ├── run_live.py      # Entry point: live trading
    └── run_backtest.py  # Entry point: backtesting
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Polymarket API credentials
```

## Strategies

| Strategy | Signal logic |
|----------|-------------|
| **MACD** | Buys bullish MACD histogram crossover, sells bearish crossover |
| **RSI** | Mean-reversion: buys oversold recovery (RSI <30→>30), sells overbought rejection |
| **CVD** | Buys bullish price/CVD divergence (price lower low, CVD higher low) and vice versa |

All strategies emit **limit orders** offset slightly from the current mid-price.

## Backtesting

```bash
python deploy/run_backtest.py \
    --data  data/my_market.csv \
    --market "0xabc..." \
    --token  "0xdef..." \
    --strategy rsi \
    --capital 1000 \
    --trade-size 50
```

CSV format:
```
timestamp,price,volume
2024-01-01T00:00:00,0.45,1200.0
2024-01-01T00:30:00,0.47,950.0
...
```

Pass custom strategy parameters as JSON:
```bash
--params '{"period": 10, "oversold": 25, "overbought": 75}'
```

## Live Trading

```bash
python deploy/run_live.py \
    --market <condition_id> \
    --token  <yes_token_id> \
    --strategy rsi \
    --capital 500
```

`--strategy` overrides `ACTIVE_STRATEGY` in `.env`.

## Risk Controls

Configured via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_POSITION_SIZE_USD` | 100 | Max USD per token |
| `MAX_TOTAL_EXPOSURE_USD` | 500 | Max total open exposure |
| `MAX_LOSS_PER_TRADE_PCT` | 0.05 | Max 5% of capital at risk per trade |
| `DAILY_LOSS_LIMIT_USD` | 200 | Halt trading if daily loss exceeds this |

## Notes

- Polymarket probability prices range from **0.01 to 0.99**
- Only **limit orders** are used — no market orders
- The bot polls the CLOB every 30 seconds (configurable in `bot/trader.py`)
- Unfilled limit orders are cancelled after `max_bars_open` ticks
- Get market/token IDs from the Polymarket CLOB API or UI
