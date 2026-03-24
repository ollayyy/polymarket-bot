#!/usr/bin/env python3
"""
Entry point: live trading.

Usage:
    python deploy/run_live.py \
        --market  <condition_id> \
        --token   <token_id> \
        --strategy rsi \
        --capital 500.0

Environment:
    Copy .env.example to .env and fill in your credentials.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Resolve project root so imports work regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from bot.risk_manager import RiskConfig
from bot.trader import Trader
from strategies import MACDStrategy, RSIStrategy, CVDStrategy

STRATEGY_MAP = {
    "macd": MACDStrategy,
    "rsi": RSIStrategy,
    "cvd": CVDStrategy,
}


def build_risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_size_usd=float(os.environ.get("MAX_POSITION_SIZE_USD", 100.0)),
        max_total_exposure_usd=float(os.environ.get("MAX_TOTAL_EXPOSURE_USD", 500.0)),
        max_loss_per_trade_pct=float(os.environ.get("MAX_LOSS_PER_TRADE_PCT", 0.05)),
        daily_loss_limit_usd=float(os.environ.get("DAILY_LOSS_LIMIT_USD", 200.0)),
    )


def main():
    parser = argparse.ArgumentParser(description="Polymarket live trading bot")
    parser.add_argument("--market", required=True, help="Polymarket condition ID")
    parser.add_argument("--token", required=True, help="YES or NO token ID")
    parser.add_argument("--strategy", default=os.environ.get("ACTIVE_STRATEGY", "rsi"), choices=list(STRATEGY_MAP))
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital in USD")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    log_file = os.environ.get("LOG_FILE")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
    )

    strategy_cls = STRATEGY_MAP[args.strategy]
    strategy = strategy_cls()
    risk_config = build_risk_config()

    trader = Trader(
        market_id=args.market,
        token_id=args.token,
        strategy=strategy,
        risk_config=risk_config,
        capital_usd=args.capital,
    )

    trader.run()


if __name__ == "__main__":
    main()
