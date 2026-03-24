"""
Risk Manager — gates every order before it reaches the exchange.

Checks performed before each order:
  1. Position size does not exceed MAX_POSITION_SIZE_USD
  2. Total open exposure does not exceed MAX_TOTAL_EXPOSURE_USD
  3. Trade does not exceed MAX_LOSS_PER_TRADE_PCT of capital
  4. Daily loss limit not breached
  5. Price is within valid probability bounds (0.01 – 0.99)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_position_size_usd: float = 100.0
    max_total_exposure_usd: float = 500.0
    max_loss_per_trade_pct: float = 0.05   # fraction of current capital
    daily_loss_limit_usd: float = 200.0
    min_price: float = 0.01
    max_price: float = 0.99


@dataclass
class OrderRequest:
    market_id: str
    token_id: str
    side: str           # "buy" or "sell"
    price: float        # limit price (0.0–1.0)
    size_usd: float


@dataclass
class RiskDecision:
    approved: bool
    adjusted_size_usd: float
    reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig, capital: float):
        self.config = config
        self.capital = capital
        self._open_positions: dict[str, float] = {}   # token_id → size_usd
        self._daily_pnl: float = 0.0
        self._day: date = datetime.utcnow().date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_order(self, req: OrderRequest) -> RiskDecision:
        """Returns an approved/adjusted order or a rejection."""
        self._roll_day_if_needed()

        # 1. Price validity
        if not (self.config.min_price <= req.price <= self.config.max_price):
            return RiskDecision(False, 0.0, f"Price {req.price} outside valid range")

        # 2. Max loss per trade
        max_loss = self.capital * self.config.max_loss_per_trade_pct
        # Worst-case loss: full size_usd goes to 0
        if req.size_usd > max_loss / req.price:
            adjusted = max_loss / req.price
            logger.warning("Reducing size from %.2f to %.2f (max-loss-per-trade)", req.size_usd, adjusted)
            req = OrderRequest(**{**req.__dict__, "size_usd": adjusted})

        # 3. Max position size
        current_pos = self._open_positions.get(req.token_id, 0.0)
        new_total = current_pos + req.size_usd
        if new_total > self.config.max_position_size_usd:
            headroom = self.config.max_position_size_usd - current_pos
            if headroom <= 0:
                return RiskDecision(False, 0.0, f"Position limit reached for {req.token_id}")
            logger.warning("Reducing size to headroom=%.2f for %s", headroom, req.token_id)
            req = OrderRequest(**{**req.__dict__, "size_usd": headroom})

        # 4. Total exposure
        total_exposure = sum(self._open_positions.values()) + req.size_usd
        if total_exposure > self.config.max_total_exposure_usd:
            return RiskDecision(False, 0.0, "Total exposure limit reached")

        # 5. Daily loss limit
        if self._daily_pnl <= -self.config.daily_loss_limit_usd:
            return RiskDecision(False, 0.0, "Daily loss limit reached — trading halted for today")

        return RiskDecision(True, req.size_usd, "OK")

    def record_fill(self, token_id: str, side: str, size_usd: float):
        """Call after a limit order fills to update exposure tracking."""
        if side == "buy":
            self._open_positions[token_id] = self._open_positions.get(token_id, 0.0) + size_usd
        elif side == "sell":
            self._open_positions[token_id] = max(
                self._open_positions.get(token_id, 0.0) - size_usd, 0.0
            )

    def record_pnl(self, pnl_usd: float):
        """Call after a position closes to update daily PnL and capital."""
        self._roll_day_if_needed()
        self._daily_pnl += pnl_usd
        self.capital += pnl_usd
        logger.info("PnL recorded: %.4f USD | daily_pnl=%.4f | capital=%.4f", pnl_usd, self._daily_pnl, self.capital)

    @property
    def total_exposure(self) -> float:
        return sum(self._open_positions.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _roll_day_if_needed(self):
        today = datetime.utcnow().date()
        if today != self._day:
            logger.info("New trading day %s — resetting daily PnL (was %.4f)", today, self._daily_pnl)
            self._daily_pnl = 0.0
            self._day = today
