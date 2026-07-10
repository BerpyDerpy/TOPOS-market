"""Regime segmentation of a RunLog (harness ground truth, INV-11).

Segments are derived from the per-step ``RegimeRecord`` log — the true
regime chain, identical across paired-seed runs (INV-9) — so every
per-regime-segment metric in this package slices identically across the
ablation conditions of one seed.
"""

from __future__ import annotations

from dataclasses import dataclass

from topos.env.harness import RunLog


@dataclass(frozen=True)
class RegimeSegment:
    """One maximal run of consecutive steps under a single true regime."""

    index: int
    regime_id: str
    start_step: int
    end_step: int
    """Exclusive: the segment covers steps [start_step, end_step)."""
    source: str
    """How the segment began: 'carry' only for the first, else
    'hazard'/'schedule' from the switch that started it."""

    @property
    def length(self) -> int:
        return self.end_step - self.start_step

    def contains(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


def regime_segments(run_log: RunLog) -> tuple[RegimeSegment, ...]:
    """Maximal constant-regime segments of the run, in step order."""
    segments: list[RegimeSegment] = []
    records = run_log.regimes
    if not records:
        return ()
    start = records[0].step
    current = records[0].regime_id
    source = records[0].source
    for record in records[1:]:
        if record.regime_id != current:
            segments.append(
                RegimeSegment(
                    index=len(segments),
                    regime_id=current,
                    start_step=start,
                    end_step=record.step,
                    source=source,
                )
            )
            start = record.step
            current = record.regime_id
            source = record.source
    segments.append(
        RegimeSegment(
            index=len(segments),
            regime_id=current,
            start_step=start,
            end_step=records[-1].step + 1,
            source=source,
        )
    )
    return tuple(segments)


def segment_of(
    segments: tuple[RegimeSegment, ...], step: int
) -> RegimeSegment | None:
    for segment in segments:
        if segment.contains(step):
            return segment
    return None


def switch_steps(segments: tuple[RegimeSegment, ...]) -> tuple[int, ...]:
    """Start steps of every segment after the first — the regime switches."""
    return tuple(segment.start_step for segment in segments[1:])
