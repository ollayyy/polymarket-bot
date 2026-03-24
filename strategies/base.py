from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    signal_type: SignalType
    market_id: str
    token_id: str
    # Probability price (0.0 - 1.0) for limit order placement
    price: float
    # Fraction of allowed position size to use (0.0 - 1.0)
    size_pct: float = 1.0
    confidence: float = 1.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.signal_type != SignalType.HOLD


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    def __init__(self, name: str, params: dict | None = None):
        self.name = name
        self.params: dict = params or {}

    @abstractmethod
    def generate_signal(
        self,
        market_id: str,
        token_id: str,
        price_series: pd.Series,
        volume_series: pd.Series | None = None,
    ) -> Signal:
        """Given OHLCV data, return a Signal."""

    def requires_volume(self) -> bool:
        """Override to True if the strategy needs volume data."""
        return False

    def min_bars(self) -> int:
        """Minimum number of bars required before a signal can be generated."""
        return 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
