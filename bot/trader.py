"""
Live Trader — executes limit orders on Polymarket via py-clob-client.

Flow per market tick:
  1. Fetch latest price/volume from CLOB
  2. Pass data to active strategy → get Signal
  3. Pass Signal through RiskManager → get approved size
  4. Place limit order via CLOB client
  5. Poll for fills; on fill, record position update
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

from strategies.base import BaseStrategy, Signal, SignalType
from bot.risk_manager import OrderRequest, RiskConfig, RiskManager

logger = logging.getLogger(__name__)

# Minimum bars of history before strategy starts generating signals
WARMUP_BARS = 60
# Polling interval in seconds between market data fetches
POLL_INTERVAL_SECONDS = 30
# Max bars of price history kept in memory
MAX_HISTORY_BARS = 500


class Trader:
    """
    Live trader for a single Polymarket market.

    Parameters
    ----------
    market_id   : Polymarket condition ID
    token_id    : YES or NO token ID
    strategy    : strategy instance
    risk_config : risk parameters
    capital_usd : starting capital in USD
    """

    def __init__(
        self,
        market_id: str,
        token_id: str,
        strategy: BaseStrategy,
        risk_config: RiskConfig,
        capital_usd: float,
    ):
        self.market_id = market_id
        self.token_id = token_id
        self.strategy = strategy
        self.risk_manager = RiskManager(risk_config, capital_usd)

        self._client: Optional[ClobClient] = None
        self._price_history: list[float] = []
        self._volume_history: list[float] = []
        self._open_order_id: Optional[str] = None
        self._open_order_side: Optional[str] = None
        self._open_order_size: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Initialise and authenticate the CLOB client.

        If POLY_API_KEY/SECRET/PASSPHRASE are set in the environment they are
        used directly. Otherwise, credentials are derived from PRIVATE_KEY via
        create_or_derive_api_creds() and printed so you can cache them in .env.
        """
        host = os.environ.get("POLY_HOST", "https://clob.polymarket.com")
        chain_id = int(os.environ.get("CHAIN_ID", "137"))
        private_key = os.environ["PRIVATE_KEY"]

        # Bootstrap client with just the wallet key to derive creds
        bootstrap = ClobClient(host=host, chain_id=chain_id, key=private_key)

        api_key = os.environ.get("POLY_API_KEY")
        if api_key:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=os.environ["POLY_API_SECRET"],
                api_passphrase=os.environ["POLY_API_PASSPHRASE"],
            )
            logger.info("Using existing API credentials from environment")
        else:
            logger.info("No API credentials found — deriving from private key...")
            raw = bootstrap.create_or_derive_api_creds()
            creds = ApiCreds(
                api_key=raw["apiKey"],
                api_secret=raw["secret"],
                api_passphrase=raw["passphrase"],
            )
            logger.info(
                "Derived credentials — add these to .env to skip derivation next time:\n"
                "  POLY_API_KEY=%s\n  POLY_API_SECRET=%s\n  POLY_API_PASSPHRASE=%s",
                raw["apiKey"], raw["secret"], raw["passphrase"],
            )

        self._client = ClobClient(
            host=host,
            chain_id=chain_id,
            key=private_key,
            creds=creds,
        )
        logger.info("Connected to Polymarket CLOB")

    def run(self):
        """Blocking main loop."""
        if self._client is None:
            self.connect()
        self._running = True
        logger.info("Trader starting on market=%s token=%s strategy=%s", self.market_id, self.token_id, self.strategy.name)
        try:
            while self._running:
                self._tick()
                time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Trader stopped by user")
        finally:
            self._cancel_open_order()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def _tick(self):
        try:
            mid_price, volume = self._fetch_market_data()
        except Exception as exc:
            logger.error("Failed to fetch market data: %s", exc)
            return

        self._price_history.append(mid_price)
        self._volume_history.append(volume)

        # Trim history
        if len(self._price_history) > MAX_HISTORY_BARS:
            self._price_history = self._price_history[-MAX_HISTORY_BARS:]
            self._volume_history = self._volume_history[-MAX_HISTORY_BARS:]

        # Check if open order was filled
        if self._open_order_id:
            self._check_fill()

        # Only generate signals after warmup and if no open order
        if len(self._price_history) < WARMUP_BARS or self._open_order_id:
            return

        price_series = pd.Series(self._price_history)
        volume_series = pd.Series(self._volume_history) if self.strategy.requires_volume() else None

        signal = self.strategy.generate_signal(
            market_id=self.market_id,
            token_id=self.token_id,
            price_series=price_series,
            volume_series=volume_series,
        )

        if signal.is_actionable():
            logger.info("Signal: %s @ %.4f — %s", signal.signal_type.value.upper(), signal.price, signal.reason)
            self._place_limit_order(signal)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _fetch_market_data(self) -> tuple[float, float]:
        """Returns (mid_price, volume) for the token."""
        book = self._client.get_order_book(self.token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.01
        best_ask = float(asks[0]["price"]) if asks else 0.99
        mid = (best_bid + best_ask) / 2.0
        # Volume approximated from top-of-book size
        volume = float(bids[0].get("size", 0)) + float(asks[0].get("size", 0)) if (bids and asks) else 0.0
        return mid, volume

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def _place_limit_order(self, signal: Signal):
        req = OrderRequest(
            market_id=self.market_id,
            token_id=self.token_id,
            side=signal.signal_type.value,
            price=signal.price,
            size_usd=100.0 * signal.size_pct,  # base size before risk check
        )
        decision = self.risk_manager.check_order(req)

        if not decision.approved:
            logger.warning("Order rejected by risk manager: %s", decision.reason)
            return

        side_const = BUY if signal.signal_type == SignalType.BUY else SELL
        order_args = OrderArgs(
            price=signal.price,
            size=decision.adjusted_size_usd,
            side=side_const,
            token_id=self.token_id,
        )

        try:
            resp = self._client.create_order(order_args)
            order_id = resp.get("orderID") or resp.get("id")
            self._open_order_id = order_id
            self._open_order_side = signal.signal_type.value
            self._open_order_size = decision.adjusted_size_usd
            logger.info(
                "Limit order placed: %s id=%s price=%.4f size=%.2f USD",
                side_const, order_id, signal.price, decision.adjusted_size_usd,
            )
        except Exception as exc:
            logger.error("Failed to place order: %s", exc)

    def _check_fill(self):
        try:
            order = self._client.get_order(self._open_order_id)
        except Exception as exc:
            logger.error("Failed to check order %s: %s", self._open_order_id, exc)
            return

        status = order.get("status", "")
        if status in ("MATCHED", "FILLED"):
            fill_price = float(order.get("avgPrice", order.get("price", 0)))
            logger.info("Order %s FILLED at %.4f", self._open_order_id, fill_price)
            self.risk_manager.record_fill(self.token_id, self._open_order_side, self._open_order_size)
            self._open_order_id = None
            self._open_order_side = None
            self._open_order_size = 0.0
        elif status in ("CANCELLED", "EXPIRED"):
            logger.info("Order %s %s — will re-signal next tick", self._open_order_id, status)
            self._open_order_id = None

    def _cancel_open_order(self):
        if self._open_order_id:
            try:
                self._client.cancel_order(self._open_order_id)
                logger.info("Cancelled open order %s on shutdown", self._open_order_id)
            except Exception as exc:
                logger.warning("Could not cancel order %s: %s", self._open_order_id, exc)
            self._open_order_id = None
