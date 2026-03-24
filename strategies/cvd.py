import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, SignalType


class CVDStrategy(BaseStrategy):
    """
    Cumulative Volume Delta (CVD) divergence strategy.

    CVD tracks net buy vs. sell volume over time. When price and CVD diverge
    (e.g., price rising but CVD falling), it often signals an impending reversal.

    Entry rules:
      - BUY  when price makes a lower low but CVD makes a higher low (bullish divergence)
      - SELL when price makes a higher high but CVD makes a lower high (bearish divergence)

    Requires trade-level data with buy/sell side information. volume_series
    should be signed: positive = buy volume, negative = sell volume.
    If unsigned volume is passed, the sign is inferred from price direction.
    """

    DEFAULT_PARAMS = {
        # Lookback window (bars) to check for divergence
        "lookback": 20,
        # Minimum price move (in probability pts) to qualify as a swing
        "min_price_swing": 0.02,
        # Minimum CVD move to qualify as a divergence
        "min_cvd_swing": 0.01,
        "limit_offset": 0.005,
        # Smoothing window for CVD
        "cvd_smooth": 3,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("CVD", merged)

    def requires_volume(self) -> bool:
        return True

    def min_bars(self) -> int:
        return self.params["lookback"] + self.params["cvd_smooth"]

    def _build_cvd(self, price_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """Compute CVD. If volume is unsigned, sign by price direction."""
        price_diff = price_series.diff()
        signed_vol = volume_series * np.sign(price_diff).fillna(0)
        cvd = signed_vol.cumsum()
        return cvd.rolling(self.params["cvd_smooth"]).mean()

    def generate_signal(
        self,
        market_id: str,
        token_id: str,
        price_series: pd.Series,
        volume_series: pd.Series | None = None,
    ) -> Signal:
        if volume_series is None or len(price_series) < self.min_bars():
            return Signal(SignalType.HOLD, market_id, token_id, price=price_series.iloc[-1])

        lookback = self.params["lookback"]
        prices = price_series.iloc[-lookback:]
        vols = volume_series.iloc[-lookback:]
        cvd = self._build_cvd(price_series, volume_series).iloc[-lookback:]

        current_price = prices.iloc[-1]
        current_cvd = cvd.iloc[-1]

        if pd.isna(current_cvd):
            return Signal(SignalType.HOLD, market_id, token_id, price=current_price)

        min_p_swing = self.params["min_price_swing"]
        min_cvd_swing = self.params["min_cvd_swing"]

        price_low = prices.min()
        price_high = prices.max()
        cvd_at_price_low = cvd.loc[prices.idxmin()] if not pd.isna(prices.idxmin()) else np.nan
        cvd_at_price_high = cvd.loc[prices.idxmax()] if not pd.isna(prices.idxmax()) else np.nan

        # Bullish divergence: price near recent low, but CVD higher than at that low
        near_low = (current_price - price_low) < min_p_swing
        if near_low and not pd.isna(cvd_at_price_low):
            cvd_divergence = current_cvd - cvd_at_price_low
            if cvd_divergence > min_cvd_swing:
                limit_price = max(current_price - self.params["limit_offset"], 0.01)
                return Signal(
                    signal_type=SignalType.BUY,
                    market_id=market_id,
                    token_id=token_id,
                    price=round(limit_price, 4),
                    confidence=min(cvd_divergence / 0.1, 1.0),
                    reason=f"CVD bullish divergence (cvd_div={cvd_divergence:.4f})",
                )

        # Bearish divergence: price near recent high, but CVD lower than at that high
        near_high = (price_high - current_price) < min_p_swing
        if near_high and not pd.isna(cvd_at_price_high):
            cvd_divergence = cvd_at_price_high - current_cvd
            if cvd_divergence > min_cvd_swing:
                limit_price = min(current_price + self.params["limit_offset"], 0.99)
                return Signal(
                    signal_type=SignalType.SELL,
                    market_id=market_id,
                    token_id=token_id,
                    price=round(limit_price, 4),
                    confidence=min(cvd_divergence / 0.1, 1.0),
                    reason=f"CVD bearish divergence (cvd_div={cvd_divergence:.4f})",
                )

        return Signal(SignalType.HOLD, market_id, token_id, price=current_price)
