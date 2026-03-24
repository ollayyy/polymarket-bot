"""
Backtest engine for limit-order strategies on Polymarket probability data.

Data format expected:
  A DataFrame with columns:
    - timestamp (index or column)
    - price     (float, 0.0–1.0)
    - volume    (float, optional — required for CVD strategy)
    - bid       (float, optional — best bid price)
    - ask       (float, optional — best ask price)

Limit order fill logic:
  - BUY  limit fills if the price drops to or below the limit price
  - SELL limit fills if the price rises to or above the limit price
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    market_id: str
    token_id: str
    side: str          # "buy" or "sell"
    entry_price: float
    exit_price: float
    size_usd: float
    entry_bar: int
    exit_bar: int
    pnl_usd: float = 0.0
    reason_in: str = ""
    reason_out: str = ""

    def __post_init__(self):
        if self.side == "buy":
            self.pnl_usd = (self.exit_price - self.entry_price) * self.size_usd
        else:
            self.pnl_usd = (self.entry_price - self.exit_price) * self.size_usd


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 1000.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl_usd > 0)
        return winners / len(self.trades)

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = pd.Series(self.equity_curve)
        rolling_max = curve.cummax()
        drawdown = (curve - rolling_max) / rolling_max
        return float(drawdown.min())

    def summary(self) -> dict:
        return {
            "trades": len(self.trades),
            "total_pnl_usd": round(self.total_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "final_equity": round(
                self.initial_capital + self.total_pnl, 4
            ),
        }


class BacktestEngine:
    """
    Event-driven backtest engine with limit-order simulation.

    Parameters
    ----------
    strategy        : strategy instance to test
    initial_capital : starting USD capital
    trade_size_usd  : fixed USD size per trade
    max_bars_open   : cancel unfilled limit order after this many bars
    fee_pct         : round-trip fee as fraction of trade value
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 1000.0,
        trade_size_usd: float = 50.0,
        max_bars_open: int = 10,
        fee_pct: float = 0.002,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.trade_size_usd = trade_size_usd
        self.max_bars_open = max_bars_open
        self.fee_pct = fee_pct

    def run(
        self,
        market_id: str,
        token_id: str,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest over the provided OHLCV DataFrame.

        `data` must have a 'price' column (0.0–1.0 probability).
        Optional 'volume' column is used by CVD strategy.
        """
        if "price" not in data.columns:
            raise ValueError("DataFrame must have a 'price' column")

        data = data.copy().reset_index(drop=True)
        price_series = data["price"]
        volume_series = data.get("volume", None)

        result = BacktestResult(initial_capital=self.initial_capital)
        equity = self.initial_capital
        result.equity_curve.append(equity)

        # Pending limit order state
        pending_signal: Optional[Signal] = None
        pending_bar: int = 0
        open_trade: Optional[dict] = None

        min_bars = self.strategy.min_bars()

        for i in range(1, len(data)):
            bar_prices = price_series.iloc[: i + 1]
            bar_volume = volume_series.iloc[: i + 1] if volume_series is not None else None
            current_price = price_series.iloc[i]

            # --- Check if open position should be closed ---
            if open_trade is not None:
                # Simple exit: reverse signal or max hold = 2 * max_bars_open
                bars_held = i - open_trade["entry_bar"]
                if bars_held >= self.max_bars_open * 2:
                    trade = Trade(
                        market_id=market_id,
                        token_id=token_id,
                        side=open_trade["side"],
                        entry_price=open_trade["entry_price"],
                        exit_price=current_price,
                        size_usd=self.trade_size_usd,
                        entry_bar=open_trade["entry_bar"],
                        exit_bar=i,
                        reason_in=open_trade["reason"],
                        reason_out="max hold period",
                    )
                    fee = self.trade_size_usd * self.fee_pct
                    equity += trade.pnl_usd - fee
                    result.trades.append(trade)
                    result.equity_curve.append(equity)
                    open_trade = None

            # --- Try to fill pending limit order ---
            if pending_signal is not None and open_trade is None:
                bars_waiting = i - pending_bar
                filled = False

                if pending_signal.signal_type == SignalType.BUY:
                    if current_price <= pending_signal.price:
                        open_trade = {
                            "side": "buy",
                            "entry_price": pending_signal.price,
                            "entry_bar": i,
                            "reason": pending_signal.reason,
                        }
                        filled = True

                elif pending_signal.signal_type == SignalType.SELL:
                    if current_price >= pending_signal.price:
                        open_trade = {
                            "side": "sell",
                            "entry_price": pending_signal.price,
                            "entry_bar": i,
                            "reason": pending_signal.reason,
                        }
                        filled = True

                if filled or bars_waiting >= self.max_bars_open:
                    pending_signal = None

            # --- Generate new signal ---
            if i >= min_bars and pending_signal is None and open_trade is None:
                signal = self.strategy.generate_signal(
                    market_id=market_id,
                    token_id=token_id,
                    price_series=bar_prices,
                    volume_series=bar_volume,
                )
                if signal.is_actionable():
                    pending_signal = signal
                    pending_bar = i
                    logger.debug("Bar %d: new %s signal @ %.4f — %s", i, signal.signal_type.value, signal.price, signal.reason)

        # Close any open position at last bar
        if open_trade is not None:
            last_price = price_series.iloc[-1]
            trade = Trade(
                market_id=market_id,
                token_id=token_id,
                side=open_trade["side"],
                entry_price=open_trade["entry_price"],
                exit_price=last_price,
                size_usd=self.trade_size_usd,
                entry_bar=open_trade["entry_bar"],
                exit_bar=len(data) - 1,
                reason_in=open_trade["reason"],
                reason_out="end of data",
            )
            fee = self.trade_size_usd * self.fee_pct
            equity += trade.pnl_usd - fee
            result.trades.append(trade)
            result.equity_curve.append(equity)

        return result
