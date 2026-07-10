"""Unit tests for the tidy-table container."""

from __future__ import annotations

import json

from topos.metrics.tables import MetricResult, TidyTable, concat


def test_from_records_column_union_and_access() -> None:
    table = TidyTable.from_records(
        "t", [{"a": 1, "b": 2.5}, {"a": 3, "c": "x"}]
    )
    assert table.columns == ("a", "b", "c")
    assert table.column("a") == [1, 3]
    assert table.column("b") == [2.5, None]
    assert len(table) == 2
    assert table.records()[1] == {"a": 3, "b": None, "c": "x"}


def test_where_filters_rows() -> None:
    table = TidyTable.from_records(
        "t",
        [
            {"condition": "FULL", "v": 1},
            {"condition": "ABL", "v": 2},
            {"condition": "FULL", "v": 3},
        ],
    )
    assert table.where(condition="FULL").column("v") == [1, 3]


def test_csv_quotes_and_markdown_truncates() -> None:
    table = TidyTable.from_records(
        "t", [{"a": 'x,"y"', "b": i} for i in range(5)]
    )
    csv = table.to_csv()
    assert csv.splitlines()[0] == "a,b"
    assert '"x,""y"""' in csv
    md = table.to_markdown(max_rows=2)
    assert "more rows" in md
    assert md.count("\n") == 4  # header + rule + 2 rows + ellipsis line


def test_metric_result_summary_is_json_serializable() -> None:
    result = MetricResult(
        name="m",
        table=TidyTable.from_records("t", [{"a": 1}]),
        summary={"FULL": {"x": 1.5}},
    )
    assert json.loads(result.summary_json()) == {"FULL": {"x": 1.5}}


def test_concat() -> None:
    t1 = TidyTable.from_records("t", [{"a": 1}])
    t2 = TidyTable.from_records("t", [{"a": 2}])
    merged = concat("all", [t1, t2])
    assert merged.column("a") == [1, 2]
    assert len(concat("empty", [])) == 0
