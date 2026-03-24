#!/usr/bin/env python3
"""
Entry point: backtesting.

Loads a CSV of historical probability price data and runs the selected
strategy through the backtest engine, then prints a results summary.

Expected CSV columns:
    timestamp, price          (required)
    volume                    (optional — needed for CVD strategy)

Usage:
    python deploy/run_backtest.py \
        --data  path/to/market_data.csv \
        --market <condition_id> \
        --token  <token_id> \
        --strategy macd \
        --capital 1000 \
        --trade-size 50

Outputs:
    Summary table + per-trade log (stdout)
    backtest_results/<strategy>_<timestamp>.csv  (trade log)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.engine import BacktestEngine
from strategies import MACDStrategy, RSIStrategy, CVDStrategy

STRATEGY_MAP = {
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "cvd": CVDStrategy,
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    if "price" not in df.columns:
        raise ValueError("CSV must contain a 'price' column (probability 0.0–1.0)")
    df["price"] = df["price"].astype(float).clip(0.001, 0.999)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df


def print_summary(result, strategy_name: str, market_id: str):
    s = result.summary()
    print("\n" + "=" * 52)
    print(f"  Backtest Results — {strategy_name.upper()}")
    print(f"  Market: {market_id}")
    print("=" * 52)
    print(f"  Trades         : {s['trades']}")
    print(f"  Win rate       : {s['win_rate']:.1%}")
    print(f"  Total PnL      : ${s['total_pnl_usd']:+.2f}")
    print(f"  Final equity   : ${s['final_equity']:.2f}")
    print(f"  Sharpe ratio   : {s['sharpe']:.3f}")
    print(f"  Max drawdown   : {s['max_drawdown']:.1%}")
    print("=" * 52 + "\n")


def save_trade_log(result, strategy_name: str):
    if not result.trades:
        return
    os.makedirs("backtest_results", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"backtest_results/{strategy_name}_{ts}.csv"
    rows = [
        {
            "market_id": t.market_id,
            "token_id": t.token_id,
            "side": t.side,
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "pnl_usd": round(t.pnl_usd, 6),
            "reason_in": t.reason_in,
            "reason_out": t.reason_out,
        }
        for t in result.trades
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trade log saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket strategy backtester")
    parser.add_argument("--data", required=True, help="Path to CSV with price history")
    parser.add_argument("--market", default="backtest-market", help="Market condition ID (label only)")
    parser.add_argument("--token", default="backtest-token", help="Token ID (label only)")
    parser.add_argument("--strategy", default="rsi", choices=list(STRATEGY_MAP))
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--trade-size", type=float, default=50.0, dest="trade_size")
    parser.add_argument("--max-bars-open", type=int, default=10, dest="max_bars_open")
    parser.add_argument("--fee-pct", type=float, default=0.002, dest="fee_pct")
    parser.add_argument("--params", default="{}", help="JSON string of strategy params")
    args = parser.parse_args()

    data = load_data(args.data)
    extra_params = json.loads(args.params)
    strategy = STRATEGY_MAP[args.strategy](params=extra_params if extra_params else None)

    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=args.capital,
        trade_size_usd=args.trade_size,
        max_bars_open=args.max_bars_open,
        fee_pct=args.fee_pct,
    )

    result = engine.run(
        market_id=args.market,
        token_id=args.token,
        data=data,
    )

    print_summary(result, args.strategy, args.market)
    save_trade_log(result, args.strategy)


if __name__ == "__main__":
    main()
