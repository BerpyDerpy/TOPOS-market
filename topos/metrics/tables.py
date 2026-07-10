"""Tiny tidy-table container for metric outputs.

Every metric returns a ``MetricResult``: one tidy table (one observation
per row, named columns) plus a JSON-serializable summary dict. A hand-rolled
container keeps the metrics package on the project's pinned dependency set
(numpy/scipy only — no pandas), while staying trivially convertible to CSV
or markdown for the ablation report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


@dataclass(frozen=True)
class TidyTable:
    """An immutable tidy table: named columns, one observation per row."""

    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    @staticmethod
    def from_records(name: str, records: Sequence[Mapping[str, Any]]) -> "TidyTable":
        """Build from a list of dicts; column order follows first appearance."""
        columns: list[str] = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        rows = tuple(
            tuple(record.get(col) for col in columns) for record in records
        )
        return TidyTable(name=name, columns=tuple(columns), rows=rows)

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list[Any]:
        idx = self.columns.index(name)
        return [row[idx] for row in self.rows]

    def records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def where(self, **conditions: Any) -> "TidyTable":
        """Rows whose named columns equal the given values."""
        indices = [self.columns.index(k) for k in conditions]
        wanted = list(conditions.values())
        rows = tuple(
            row
            for row in self.rows
            if all(row[i] == v for i, v in zip(indices, wanted))
        )
        return TidyTable(name=self.name, columns=self.columns, rows=rows)

    def to_csv(self) -> str:
        def cell(value: Any) -> str:
            text = "" if value is None else str(value)
            if any(ch in text for ch in ",\"\n"):
                text = '"' + text.replace('"', '""') + '"'
            return text

        lines = [",".join(self.columns)]
        lines.extend(",".join(cell(v) for v in row) for row in self.rows)
        return "\n".join(lines) + "\n"

    def to_markdown(self, max_rows: int | None = None) -> str:
        header = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join(" --- " for _ in self.columns) + "|"
        body_rows = self.rows if max_rows is None else self.rows[:max_rows]
        body = [
            "| " + " | ".join(_fmt(v) for v in row) + " |" for row in body_rows
        ]
        if max_rows is not None and len(self.rows) > max_rows:
            body.append(
                f"| ... {len(self.rows) - max_rows} more rows ... |"
            )
        return "\n".join([header, rule, *body])


@dataclass(frozen=True)
class MetricResult:
    """One metric's full output: tidy table + serializable summary."""

    name: str
    table: TidyTable
    summary: dict[str, Any] = field(default_factory=dict)

    def summary_json(self) -> str:
        return json.dumps(self.summary, indent=2, sort_keys=True, default=str)


def concat(name: str, tables: Iterable[TidyTable]) -> TidyTable:
    """Concatenate tables with identical columns (in any column order)."""
    materialized = [t for t in tables if len(t)]
    if not materialized:
        return TidyTable(name=name, columns=(), rows=())
    columns = materialized[0].columns
    records: list[dict[str, Any]] = []
    for table in materialized:
        records.extend(table.records())
    return TidyTable.from_records(name, records)
