"""Broadcast conditioning: the "global broadcast" half of GWT, made real.

Winning the salience competition must MEAN something computationally, or
the workspace is decoration. The mechanism: after the competition and
before any module's next update, the current ``Focus`` (or None) is
pushed to every registered consumer through one hook:

    def condition_on_focus(self, focus: Focus | None) -> None: ...

Each consumer compares the focus to its own hypothesis id and adjusts the
GRANULARITY of its next work accordingly — never the correctness of its
posteriors. The two wired exemplars (P9):

* ``FlowIntensity`` refreshes its per-band Gamma posteriors only when
  focused; unfocused, it buffers counts and maintains a coarse aggregate
  posterior over the total rate. Because Gamma-Poisson batches exactly
  (sum the counts, sum the exposure), the fine posteriors catch up
  EXACTLY when focus arrives: attention chooses the resolution of the
  live representation, conservation of evidence guarantees nothing is
  ever lost.
* ``FairValueKF`` runs its parameter-EIG quadrature only when focused;
  unfocused, curiosity is quoted from the last focused refresh (a stale
  but honest quote — the quadrature is the work attention pays for).

The pattern for P12 (and any future module): implement the hook, treat
"focused" as permission to spend, and NEVER let focus change what the
posterior would eventually converge to — only when the fine-grained
version of it is materialized. Register every hook-bearing module in
``Workspace(consumers=...)``; registration is explicit and validated so a
missing wire fails loudly instead of silently degrading to a decorative
broadcast.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from topos.contracts.workspace import Focus


@runtime_checkable
class FocusConsumer(Protocol):
    """Anything that conditions its next work on the broadcast focus."""

    def condition_on_focus(self, focus: Focus | None) -> None:
        """Receive the cycle's focus (None on quiet cycles)."""
        ...


def validate_consumers(consumers: Iterable[object]) -> tuple[FocusConsumer, ...]:
    """Fail loudly at construction if any registered consumer lacks the hook."""
    validated: list[FocusConsumer] = []
    for consumer in consumers:
        if not isinstance(consumer, FocusConsumer):
            raise TypeError(
                f"{consumer!r} is registered for broadcast conditioning but "
                "has no condition_on_focus(focus) method"
            )
        validated.append(consumer)
    return tuple(validated)


def broadcast_focus(
    consumers: Sequence[FocusConsumer], focus: Focus | None
) -> None:
    """Push the cycle's focus to every registered consumer.

    Called exactly once per cycle, after the salience competition and
    before the arbitration that may trigger further module work — so the
    focused module is already conditioned when the proposer's refined
    menu interrogates it, and every module is conditioned before its next
    ``update``.
    """
    for consumer in consumers:
        consumer.condition_on_focus(focus)
