"""Wire-level contracts between the agent and the exchange environment.

All prices are integer ticks and all sizes are integer lots, everywhere.
These types are the ONLY things that cross the agent/environment boundary:
the agent sends `ExchangeMessage`s, the environment answers with a single
`Observation` per step and nothing else (INV-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto, unique
from typing import Final, TypeAlias

N_LEVELS: Final[int] = 10
"""Book levels per side in every Observation, best price first."""

GTC: Final[int] = 0
"""`PlaceLimit.tif_steps == GTC` (0) means good-till-cancel."""


@unique
class Side(IntEnum):
    """Order/trade direction; the integer value is the price-direction sign."""

    BUY = 1
    SELL = -1


@unique
class AckStatus(Enum):
    ACCEPTED = auto()
    REJECTED = auto()
    CANCELED = auto()
    EXPIRED = auto()


@unique
class Liquidity(Enum):
    MAKER = auto()
    TAKER = auto()


@dataclass(frozen=True)
class PlaceLimit:
    """Place a limit order.

    `tif_steps == 0` means good-till-cancel; otherwise the order expires
    (with an EXPIRED ack) after `tif_steps` engine steps.
    """

    side: Side
    price_ticks: int
    size_lots: int
    tif_steps: int

    def __post_init__(self) -> None:
        if self.size_lots <= 0:
            raise ValueError(f"size_lots must be positive, got {self.size_lots}")
        if self.tif_steps < 0:
            raise ValueError(f"tif_steps must be >= 0 (0 => GTC), got {self.tif_steps}")


@dataclass(frozen=True)
class Cancel:
    order_id: int


ExchangeMessage: TypeAlias = PlaceLimit | Cancel


@dataclass(frozen=True)
class Ack:
    order_id: int
    status: AckStatus
    step: int


@dataclass(frozen=True)
class Fill:
    order_id: int
    price_ticks: int
    size_lots: int
    liquidity: Liquidity
    step: int


@dataclass(frozen=True)
class BookLevel:
    """One visible price level; `size_lots == 0` marks an empty/padded level."""

    price_ticks: int
    size_lots: int


@dataclass(frozen=True)
class Trade:
    """A public trade print observed since the previous observation."""

    price_ticks: int
    size_lots: int
    aggressor: Side


@dataclass(frozen=True)
class Observation:
    """Everything the agent is allowed to see, once per step.

    Deliberately absent, and to stay absent — do not add fields (enforced by
    tests/tripwires/test_observation_shape.py):

      * any scalar feedback signal (INV-1),
      * engine-side account state (INV-11),
      * ground-truth queue position (INV-11),
      * any information about other agents (INV-11).

    Those quantities flow only through harness-only channels into
    metrics/validation, never into agent-facing code.
    """

    step: int
    bids: tuple[BookLevel, ...]
    """Exactly N_LEVELS levels, best (highest) price first; pad with size 0."""
    asks: tuple[BookLevel, ...]
    """Exactly N_LEVELS levels, best (lowest) price first; pad with size 0."""
    trades: tuple[Trade, ...]
    """Public trade prints since the last observation."""
    own_acks: tuple[Ack, ...]
    own_fills: tuple[Fill, ...]

    def __post_init__(self) -> None:
        if len(self.bids) != N_LEVELS or len(self.asks) != N_LEVELS:
            raise ValueError(
                f"bids/asks must each hold exactly {N_LEVELS} levels "
                f"(pad empties with size_lots=0); got "
                f"{len(self.bids)} bids / {len(self.asks)} asks"
            )
