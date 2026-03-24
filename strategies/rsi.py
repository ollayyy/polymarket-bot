import pandas as pd
import ta

from .base import BaseStrategy, Signal, SignalType


class RSIStrategy(BaseStrategy):
    """
    RSI mean-reversion strategy for prediction market probability prices.

    Entry rules:
      - BUY  when RSI crosses up through oversold threshold (default 30)
      - SELL when RSI crosses down through overbought threshold (default 70)

    Particularly effective on prediction markets where prices often overshoot
    fundamental probability in either direction due to momentum chasing.
    """

    DEFAULT_PARAMS = {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        # Minimum RSI exit magnitude to filter weak signals
        "min_rsi_move": 2.0,
        "limit_offset": 0.005,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("RSI", merged)

    def min_bars(self) -> int:
        return self.params["period"] + 1

    def generate_signal(
        self,
        market_id: str,
        token_id: str,
        price_series: pd.Series,
        volume_series: pd.Series | None = None,
    ) -> Signal:
        if len(price_series) < self.min_bars():
            return Signal(SignalType.HOLD, market_id, token_id, price=price_series.iloc[-1])

        rsi = ta.momentum.RSIIndicator(close=price_series, window=self.params["period"]).rsi()

        if rsi.isna().iloc[-1]:
            return Signal(SignalType.HOLD, market_id, token_id, price=price_series.iloc[-1])

        prev_rsi = rsi.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        current_price = price_series.iloc[-1]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]
        min_move = self.params["min_rsi_move"]

        # Crossing up through oversold → mean reversion buy
        if prev_rsi < oversold and curr_rsi >= oversold and (curr_rsi - prev_rsi) >= min_move:
            limit_price = max(current_price - self.params["limit_offset"], 0.01)
            confidence = (curr_rsi - prev_rsi) / 10.0
            return Signal(
                signal_type=SignalType.BUY,
                market_id=market_id,
                token_id=token_id,
                price=round(limit_price, 4),
                confidence=min(confidence, 1.0),
                reason=f"RSI oversold recovery ({prev_rsi:.1f} → {curr_rsi:.1f})",
            )

        # Crossing down through overbought → mean reversion sell
        if prev_rsi > overbought and curr_rsi <= overbought and (prev_rsi - curr_rsi) >= min_move:
            limit_price = min(current_price + self.params["limit_offset"], 0.99)
            confidence = (prev_rsi - curr_rsi) / 10.0
            return Signal(
                signal_type=SignalType.SELL,
                market_id=market_id,
                token_id=token_id,
                price=round(limit_price, 4),
                confidence=min(confidence, 1.0),
                reason=f"RSI overbought rejection ({prev_rsi:.1f} → {curr_rsi:.1f})",
            )

        return Signal(SignalType.HOLD, market_id, token_id, price=current_price)
