"""CSV/XLSX ingestion, profiling, allowlisted analysis and charts (real pandas, no model)."""

import io
import uuid
import zipfile

import pandas as pd
import pytest
from pydantic import ValidationError

from omniflow.core.config import Settings
from omniflow.core.sessions import SessionStore
from omniflow.data.analysis import AnalysisArgs, AnalysisError
from omniflow.data.charts import ChartArgs
from omniflow.data.ingest import DataError, Limits, load_upload
from omniflow.data.service import DataService
from omniflow.ui.artifacts import safe_csv
from tests.data_fixtures import SMALL_CSV, sales_frame, to_csv_bytes, to_xlsx_bytes

LIMITS = Limits(max_bytes=5_000_000, max_rows=10_000, max_columns=50, max_memory_mb=100,
                max_sheets=3)


def _service(**overrides) -> DataService:
    settings = Settings(_env_file=None, **overrides)
    return DataService(settings, SessionStore(dict, max_sessions=10, ttl_seconds=3600))


def _loaded(data=SMALL_CSV, name="small.csv", **overrides):
    service = _service(**overrides)
    sid = uuid.uuid4().hex
    report = service.ingest(sid, name, data)
    assert report.status == "loaded", report.message
    return service, sid


def _run(service, sid, **kwargs):
    return service.analyze(sid, AnalysisArgs(dataset=kwargs.pop("dataset", "small.csv"),
                                             **kwargs))[1]


# ------------------------------------------------------------------ ingestion
def test_csv_ingestion_types_and_dimensions():
    table = load_upload("small.csv", SMALL_CSV, LIMITS)[0]
    assert table.frame.shape == (6, 6)
    assert pd.api.types.is_numeric_dtype(table.frame["units"])
    assert pd.api.types.is_datetime64_any_dtype(table.frame["order_date"])
    assert table.frame["units"].isna().sum() == 1


def test_xlsx_ingestion_with_multiple_sheets():
    data = to_xlsx_bytes({"Q1": sales_frame(20), "Q2": sales_frame(10), "Empty": pd.DataFrame()})
    tables = load_upload("book.xlsx", data, LIMITS)
    assert [t.name for t in tables] == ["book.xlsx [Q1]", "book.xlsx [Q2]"]
    assert tables[0].frame.shape == (21, 7)


@pytest.mark.parametrize(
    "name,data,code",
    [
        ("data.txt", b"a,b\n1,2", "unsupported_type"),
        ("macro.xlsm", b"PK\x03\x04", "unsupported_type"),
        ("old.xls", b"\xd0\xcf\x11\xe0", "unsupported_type"),
        ("empty.csv", b"", "empty_file"),
        ("binary.csv", b"\x00\x01\x02binary", "unsupported_type"),
        ("fake.xlsx", b"not a zip at all", "unsupported_type"),
        ("broken.xlsx", b"PK\x03\x04garbage", "malformed"),
        ("header_only.csv", b"a,b,c\n", "empty_table"),
    ],
)
def test_bad_files_are_rejected(name, data, code):
    with pytest.raises(DataError) as err:
        load_upload(name, data, LIMITS)
    assert err.value.code == code


def test_size_row_and_column_limits():
    with pytest.raises(DataError, match="larger than"):
        load_upload("big.csv", b"a\n" + b"1\n" * 100, Limits(50, 10, 10, 10, 1))
    with pytest.raises(DataError) as err:
        load_upload("rows.csv", to_csv_bytes(sales_frame(60)), Limits(10**7, 50, 50, 100, 1))
    assert err.value.code == "too_many_rows"
    wide = pd.DataFrame([list(range(60))], columns=[f"c{i}" for i in range(60)])
    with pytest.raises(DataError) as err:
        load_upload("wide.csv", to_csv_bytes(wide), LIMITS)
    assert err.value.code == "too_many_columns"


def test_zip_bomb_and_macro_workbooks_are_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/vbaProject.bin", b"macro")
    with pytest.raises(DataError, match="macros"):
        load_upload("m.xlsx", buffer.getvalue(), LIMITS)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/huge.xml", b"0" * 260_000_000)
    with pytest.raises(DataError, match="expands"):
        load_upload("bomb.xlsx", buffer.getvalue(), Limits(10**8, 10, 10, 10, 1))


def test_duplicate_blank_columns_encoding_and_bad_numbers():
    data = "id,Amount,Amount,,City\n1,10,1,x,Zürich\n2,abc,2,y,Köln\n3,30,3,z,Paris\n".encode(
        "cp1252")
    table = load_upload("messy.csv", data, LIMITS)[0]
    assert list(table.frame.columns) == ["id", "Amount", "Amount_2", "column_4", "City"]
    assert "Zürich" in table.frame["City"].tolist()  # decoded via the Windows-1252 fallback
    joined = " ".join(table.warnings)
    assert "Duplicate column names" in joined and "Windows-1252" in joined
    assert table.frame["Amount"].dtype == object  # 1 of 3 non-numeric: stays text, no guessing


def test_malformed_csv_lines_are_skipped_with_a_warning():
    data = b"a,b\n1,2\n3,4,5,6\n7,8\n"
    table = load_upload("bad.csv", data, LIMITS)[0]
    assert len(table.frame) == 2
    assert any("malformed" in w for w in table.warnings)


def test_formula_cells_are_plain_data_and_csv_export_neutralises_them():
    data = b"name,note\nalice,=HYPERLINK(\"http://evil\")\nbob,+cmd|' /C calc'!A0\n"
    table = load_upload("formula.csv", data, LIMITS)[0]
    assert table.frame["note"][0].startswith("=HYPERLINK")  # stored as text, never evaluated
    exported = safe_csv({"columns": ["note"], "rows": [[v] for v in table.frame["note"]]})
    assert b"'=HYPERLINK" in exported and b"'+cmd" in exported


# -------------------------------------------------------------------- profile
def test_profile_counts_missing_duplicates_outliers_and_identifiers():
    service, sid = _loaded(to_csv_bytes(sales_frame()), "sales.csv")
    d = service.find(sid, "sales.csv")
    p = d.profile
    assert p["rows"] == 121 and p["duplicate_rows"] == 1
    assert "order_id" in p["identifier_columns"]
    revenue = next(c for c in p["column_profiles"] if c["name"] == "revenue")
    assert revenue["role"] == "numeric" and revenue["outliers"]["count"] >= 1
    assert all("order_id" not in (c["a"], c["b"]) for c in p["correlations"])


# ------------------------------------------------------------------- analysis
def test_describe_and_missing_values_are_correct():
    service, sid = _loaded()
    out = _run(service, sid, operation="describe", columns=["units"])
    row = dict(zip(out["table"]["columns"], out["table"]["rows"][0], strict=True))
    assert row["count"] == 5 and row["mean"] == pytest.approx(4.4) and row["max"] == 10
    out = _run(service, sid, operation="missing_values")
    assert out["table"]["rows"][0][:2] == ["units", 1]


def test_group_aggregate_with_filter():
    service, sid = _loaded()
    out = _run(service, sid, operation="group_aggregate", group_by=["region"],
               value_column="units", aggregation="sum")
    assert dict(out["table"]["rows"]) == {"North": 11.0, "South": 9.0, "East": 2.0}
    out = _run(service, sid, operation="group_aggregate", group_by=["product"],
               value_column="units", aggregation="count",
               filters=[{"column": "region", "op": "==", "value": "North"}])
    assert dict(out["table"]["rows"]) == {"Widget": 1, "Gadget": 1}  # missing units not counted
    assert "3 of 6 rows" in out["notes"][0]


def test_value_counts_compare_groups_crosstab_and_top_rows():
    service, sid = _loaded()
    out = _run(service, sid, operation="value_counts", columns=["product"])
    assert out["table"]["rows"][0][:2] == ["Widget", 3]
    out = _run(service, sid, operation="compare_groups", group_by=["product"],
               value_column="unit_price")
    assert out["table"]["rows"][0][0] == "Gadget"
    out = _run(service, sid, operation="crosstab", group_by=["region", "product"])
    assert "Widget" in out["table"]["columns"]
    out = _run(service, sid, operation="top_rows", value_column="unit_price", top_n=2)
    assert len(out["table"]["rows"]) == 2


def test_time_trend_by_month():
    service, sid = _loaded()
    out = _run(service, sid, operation="time_trend", date_column="order_date",
               value_column="units", aggregation="sum", frequency="month")
    assert [r[1] for r in out["table"]["rows"]] == [14.0, 3.0, 5.0]
    assert out["metrics"]["change"] == -9.0


def test_correlation_and_outliers_on_larger_data():
    service, sid = _loaded(to_csv_bytes(sales_frame()), "sales.csv")
    out = _run(service, sid, dataset="sales.csv", operation="outliers", value_column="revenue")
    assert out["metrics"]["outlier_count"] >= 1
    revenue = out["table"]["columns"].index("revenue")
    assert 99999.0 in [row[revenue] for row in out["table"]["rows"]]
    out = _run(service, sid, dataset="sales.csv", operation="correlation")
    assert "Correlation shows association" in out["notes"][0]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"operation": "group_aggregate", "group_by": ["nope"], "value_column": "units"},
         "Unknown column"),
        ({"operation": "group_aggregate", "group_by": ["region"], "value_column": "region"},
         "not numeric"),
        ({"operation": "group_aggregate", "group_by": ["region"], "value_column": "order_id"},
         "identifier"),
        ({"operation": "time_trend", "date_column": "region"}, "not a date"),
        ({"operation": "crosstab", "group_by": ["region"]}, "exactly two"),
        ({"operation": "correlation"}, "Too few rows"),
        ({"operation": "group_aggregate", "group_by": ["region"], "value_column": "units",
          "filters": [{"column": "units", "op": ">", "value": "lots"}]}, "must be a number"),
    ],
)
def test_invalid_analysis_requests_are_explained(kwargs, message):
    service, sid = _loaded()
    with pytest.raises(AnalysisError, match=message):
        _run(service, sid, **kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        {"operation": "exec"},
        {"operation": "describe", "code": "__import__('os').system('calc')"},
        {"operation": "describe", "top_n": 10_000},
        {"operation": "group_aggregate", "aggregation": "eval"},
    ],
)
def test_arbitrary_code_and_unknown_operations_are_rejected_by_schema(bad):
    with pytest.raises(ValidationError):
        AnalysisArgs(dataset="small.csv", **bad)


def test_code_like_column_names_are_just_data():
    data = b"__import__('os').system('calc'),value\n1,2\n3,4\n"
    service, sid = _loaded(data, "evil.csv")
    with pytest.raises(AnalysisError, match="Unknown column"):
        _run(service, sid, dataset="evil.csv", operation="describe", columns=["os.environ"])
    out = _run(service, sid, dataset="evil.csv", operation="describe", columns=["value"])
    assert out["table"]["rows"]


# --------------------------------------------------------------------- charts
@pytest.mark.parametrize(
    "args",
    [
        {"chart_type": "bar", "x": "product", "y": "units"},
        {"chart_type": "bar", "x": "region"},
        {"chart_type": "line", "x": "order_date", "y": "units"},
        {"chart_type": "histogram", "x": "unit_price", "bins": 5},
    ],
)
def test_charts_are_built_from_computed_data(args):
    service, sid = _loaded()
    _, chart = service.chart(sid, ChartArgs(dataset="small.csv", **args))
    assert chart["data"] and chart["type"] == args["chart_type"]
    from omniflow.ui.artifacts import chart_object

    assert chart_object(chart).to_dict()["mark"]  # renders to a valid Vega-Lite spec


def test_scatter_is_sampled_and_heatmap_needs_data():
    service, sid = _loaded(to_csv_bytes(sales_frame()), "sales.csv", data_chart_points=50)
    _, chart = service.chart(sid, ChartArgs(dataset="sales.csv", chart_type="scatter",
                                            x="units", y="revenue"))
    assert len(chart["data"]) == 50 and "sample" in chart["notes"][0]
    _, heat = service.chart(sid, ChartArgs(dataset="sales.csv", chart_type="heatmap"))
    assert {d["x"] for d in heat["data"]} >= {"units", "revenue"}
    service2, sid2 = _loaded()
    with pytest.raises(AnalysisError):
        service2.chart(sid2, ChartArgs(dataset="small.csv", chart_type="heatmap"))


@pytest.mark.parametrize(
    "args,message",
    [
        ({"chart_type": "scatter", "x": "units"}, "needs numeric x and y"),
        ({"chart_type": "histogram", "x": "region"}, "not numeric"),
        ({"chart_type": "bar"}, "needs an x column"),
        ({"chart_type": "bar", "x": "missing_col"}, "Unknown column"),
    ],
)
def test_invalid_charts_are_explained(args, message):
    service, sid = _loaded()
    with pytest.raises(AnalysisError, match=message):
        service.chart(sid, ChartArgs(dataset="small.csv", **args))


# ------------------------------------------------------- sessions and limits
def test_datasets_are_isolated_per_session_and_clearing_is_scoped():
    service = _service()
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    service.ingest(a, "small.csv", SMALL_CSV)
    service.ingest(b, "sales.csv", to_csv_bytes(sales_frame()))
    assert [d.name for d in service.datasets(a)] == ["small.csv"]
    with pytest.raises(AnalysisError, match="Unknown or ambiguous"):
        service.find(b, "small.csv")
    service.clear(a)
    assert service.datasets(a) == [] and len(service.datasets(b)) == 1


def test_duplicates_file_limit_and_invalid_session():
    service = _service(data_max_files=1)
    sid = uuid.uuid4().hex
    assert service.ingest(sid, "small.csv", SMALL_CSV).status == "loaded"
    assert service.ingest(sid, "copy.csv", SMALL_CSV).status == "duplicate"
    assert "limit is 1" in service.ingest(sid, "sales.csv", to_csv_bytes(sales_frame())).message
    assert service.ingest("../../etc", "small.csv", SMALL_CSV).status == "rejected"


def test_unexpected_errors_are_hidden(monkeypatch):
    from omniflow.data import service as service_module

    def boom(*_a, **_k):
        raise RuntimeError("C:/secret/path exploded")

    monkeypatch.setattr(service_module, "load_upload", boom)
    report = _service().ingest(uuid.uuid4().hex, "small.csv", SMALL_CSV)
    assert report.status == "rejected" and "secret" not in report.message


def test_charts_handle_column_names_that_look_like_encoding_syntax():
    data = b"Sales.Q1:N,[Region]\n10,a\n20,b\n5,a\n"
    service, sid = _loaded(data, "odd.csv")
    from omniflow.ui.artifacts import chart_object

    _, chart = service.chart(sid, ChartArgs(dataset="odd.csv", chart_type="bar",
                                            x="[Region]", y="Sales.Q1:N"))
    spec = chart_object(chart).to_dict()
    assert spec["encoding"]["x"]["field"] == "x"
    assert spec["encoding"]["x"]["title"] == "[Region]"
