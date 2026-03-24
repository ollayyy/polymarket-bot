import pandas as pd
import ta

from .base import BaseStrategy, Signal, SignalType


class MACDStrategy(BaseStrategy):
    """
    MACD crossover strategy adapted for prediction-market probability prices.

    Entry rules:
      - BUY  when MACD line crosses above signal line (bullish crossover)
      - SELL when MACD line crosses below signal line (bearish crossover)

    Limit order price is offset slightly from current mid to improve fill rate.
    """

    DEFAULT_PARAMS = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        # How far inside the spread to place the limit order (in probability pts)
        "limit_offset": 0.005,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("MACD", merged)

    def min_bars(self) -> int:
        return self.params["slow_period"] + self.params["signal_period"]

    def generate_signal(
        self,
        market_id: str,
        token_id: str,
        price_series: pd.Series,
        volume_series: pd.Series | None = None,
    ) -> Signal:
        if len(price_series) < self.min_bars():
            return Signal(SignalType.HOLD, market_id, token_id, price=price_series.iloc[-1])

        macd_indicator = ta.trend.MACD(
            close=price_series,
            window_fast=self.params["fast_period"],
            window_slow=self.params["slow_period"],
            window_sign=self.params["signal_period"],
        )

        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()

        if macd_line.isna().iloc[-1] or signal_line.isna().iloc[-1]:
            return Signal(SignalType.HOLD, market_id, token_id, price=price_series.iloc[-1])

        prev_hist = histogram.iloc[-2]
        curr_hist = histogram.iloc[-1]
        current_price = price_series.iloc[-1]

        # Bullish crossover: histogram flipped positive
        if prev_hist <= 0 < curr_hist:
            limit_price = min(current_price + self.params["limit_offset"], 0.99)
            return Signal(
                signal_type=SignalType.BUY,
                market_id=market_id,
                token_id=token_id,
                price=round(limit_price, 4),
                confidence=abs(curr_hist),
                reason=f"MACD bullish crossover (hist={curr_hist:.4f})",
            )

        # Bearish crossover: histogram flipped negative
        if prev_hist >= 0 > curr_hist:
            limit_price = max(current_price - self.params["limit_offset"], 0.01)
            return Signal(
                signal_type=SignalType.SELL,
                market_id=market_id,
                token_id=token_id,
                price=round(limit_price, 4),
                confidence=abs(curr_hist),
                reason=f"MACD bearish crossover (hist={curr_hist:.4f})",
            )

        return Signal(SignalType.HOLD, market_id, token_id, price=current_price)
