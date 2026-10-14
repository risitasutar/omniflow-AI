"""Session-isolated "Import CSV/XLSX into SQL": safe names, parameterised inserts,
querying through the read-only path, isolation, limits, cleanup and security."""

import hashlib
import uuid

import pandas as pd
import pytest

from omniflow.core.config import Settings
from omniflow.core.sessions import SessionStore
from omniflow.data.ingest import Limits, load_upload
from omniflow.sql.database import SqlExecutionError
from omniflow.sql.guard import SqlRejected
from omniflow.sql.importer import normalize_identifier
from omniflow.sql.service import IMPORTED_NAME, SAMPLE_NAME, SqlService
from omniflow.tools import build_default_registry
from omniflow.tools.registry import ToolContext, ToolError
from omniflow.tools.sql_tools import SqlQueryArgs, SqlSchemaArgs, make_sql_tools
from tests.data_fixtures import SMALL_CSV, to_xlsx_bytes

LIMITS = Limits(10_000_000, 10_000, 50, 100, 3)
BLINKIT = "BlinkIT Grocery Data (1).csv"


@pytest.fixture
def service(tmp_path):
    return SqlService(Settings(_env_file=None, sql_sample_path=tmp_path / "shop.db"))


def _frame(data=SMALL_CSV, name="small.csv"):
    return load_upload(name, data, LIMITS)[0].frame


def _import(service, sid=None, frame=None, name=BLINKIT):
    sid = sid or uuid.uuid4().hex
    report = service.import_frame(sid, _frame() if frame is None else frame, name, name)
    return sid, report


# ---------------------------------------------------------------- naming
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BlinkIT Grocery Data (1)", "blinkit_grocery_data_1"),
        ("Sales 2025!!", "sales_2025"),
        ("2025 sales", "t_2025_sales"),
        ("select", "t_select"),
        ("sqlite_master", "t_sqlite_master"),
        ('x"; DROP TABLE orders; --', "x_drop_table_orders"),
        ("", "fallback"),
        ("Ünïcödé ✓", "n_c_d"),
    ],
)
def test_identifiers_are_normalised(raw, expected):
    assert normalize_identifier(raw, "fallback") == expected


# -------------------------------------------------------------- importing
def test_csv_import_creates_table_with_schema_and_rows(service):
    sid, report = _import(service)
    assert report.table == "blinkit_grocery_data_1"
    assert report.rows == 6
    types = {c.sql_name: c.sql_type for c in report.columns}
    assert types["order_id"] == "INTEGER" and types["unit_price"] == "REAL"
    assert types["region"] == "TEXT" and types["order_date"] == "TEXT"
    schema = service.schema(IMPORTED_NAME, sid)
    assert schema[0]["table"] == report.table and schema[0]["rows"] == 6


def test_xlsx_sheets_import_as_separate_tables(service):
    frame = pd.DataFrame({"product": ["A", "B"], "units": [3, 4]})
    tables = load_upload("book.xlsx", to_xlsx_bytes({"Jan": frame, "Feb": frame}), LIMITS)
    sid = uuid.uuid4().hex
    names = [service.import_frame(sid, t.frame, t.name, t.name).table for t in tables]
    assert names == ["book_jan", "book_feb"]


def test_imported_data_is_queryable_with_correct_values(service):
    sid, report = _import(service)
    out = service.query(IMPORTED_NAME, f"SELECT region, SUM(units) AS u FROM {report.table} "
                        "GROUP BY region ORDER BY u DESC", sid)
    assert out["rows"] == [["North", 11.0], ["South", 9.0], ["East", 2.0]]
    assert out["imported"] is True and out["sample_data"] is False


def test_values_are_inserted_as_data_not_sql(service):
    hostile = pd.DataFrame({"note": ["'); DROP TABLE x; --", '" OR 1=1 --', "=CMD()"],
                            "n": [1, 2, 3]})
    sid, report = _import(service, frame=hostile, name="hostile")
    out = service.query(IMPORTED_NAME, f"SELECT note FROM {report.table} ORDER BY n", sid)
    assert [r[0] for r in out["rows"]] == list(hostile["note"])


def test_duplicate_table_names_get_a_suffix(service):
    sid, first = _import(service)
    _, second = _import(service, sid=sid)
    assert (first.table, second.table) == ("blinkit_grocery_data_1", "blinkit_grocery_data_1_2")


def test_import_limits(tmp_path):
    service = SqlService(Settings(_env_file=None, sql_sample_path=tmp_path / "s.db",
                                  sql_import_max_tables=1, data_max_rows=10))
    with pytest.raises(SqlRejected, match="more than 10 rows"):
        _import(service, frame=pd.concat([_frame(), _frame()]))  # 12 rows
    sid, _ = _import(service, frame=_frame().head(2), name="a")
    with pytest.raises(SqlRejected, match="limit is 1"):
        _import(service, sid=sid, frame=_frame().head(2), name="b")
    with pytest.raises(SqlRejected, match="no rows"):
        _import(service, frame=pd.DataFrame(), name="empty")
    with pytest.raises(SqlRejected, match="not valid"):
        service.import_frame("../../evil", _frame(), "x", "x")


def test_importing_does_not_change_the_source_dataset(service):
    frame = _frame()
    before = frame.copy()
    _import(service, frame=frame)
    pd.testing.assert_frame_equal(frame, before)


# ------------------------------------------------------- isolation/cleanup
def test_sessions_cannot_see_each_others_imports(service):
    sid_a, report = _import(service)
    sid_b = uuid.uuid4().hex
    assert [d.name for d in service.databases(session_id=sid_b)] == [SAMPLE_NAME]
    with pytest.raises(SqlRejected, match="Unknown database"):
        service.query(IMPORTED_NAME, f"SELECT * FROM {report.table}", sid_b)
    with pytest.raises(SqlRejected, match="Unknown database"):
        service.query(IMPORTED_NAME, f"SELECT * FROM {report.table}")  # no session at all


def test_clearing_imports_deletes_the_session_file_only(service):
    sid_a, _ = _import(service)
    sid_b, _ = _import(service)
    path_a = service.imported_database(sid_a).path
    path_b = service.imported_database(sid_b).path
    service.clear_imports(sid_a)
    assert not path_a.exists() and path_b.exists()
    assert IMPORTED_NAME not in [d.name for d in service.databases(session_id=sid_a)]


def test_idle_sessions_expire_and_their_files_are_deleted():
    now = [0.0]
    deleted = []
    store = SessionStore(dict, max_sessions=5, ttl_seconds=10, clock=lambda: now[0],
                         on_drop=lambda sid, value: deleted.append(sid))
    sid = uuid.uuid4().hex
    store.get(sid, create=True)["db"] = "x"
    now[0] = 100
    assert store.get(sid) is None and deleted == [sid]


def test_import_files_live_outside_the_repository(service):
    sid, _ = _import(service)
    path = service.imported_database(sid).path
    assert "campusx-agent-project" not in str(path).replace("\\", "/")
    assert path.name == f"{sid}.db"


# ----------------------------------------------------------------- security
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO {t} VALUES (1)", "UPDATE {t} SET units = 0", "DELETE FROM {t}",
        "DROP TABLE {t}", "ALTER TABLE {t} ADD COLUMN x", "CREATE TABLE evil (x)",
        "ATTACH DATABASE 'x.db' AS x", "PRAGMA table_info({t})",
        "SELECT * FROM {t}; DROP TABLE {t}", "SELECT * FROM sqlite_master",
        "SELECT * FROM orders", "SELECT load_extension('x') FROM {t}",
    ],
)
def test_imported_database_keeps_read_only_protections(service, sql):
    sid, report = _import(service)
    path = service.imported_database(sid).path
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((SqlRejected, SqlExecutionError)):
        service.query(IMPORTED_NAME, sql.format(t=report.table), sid)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_no_tool_can_import_or_write():
    registry = build_default_registry(Settings(_env_file=None, sql_sample_enabled=False),
                                      sql=SqlService(Settings(_env_file=None,
                                                              sql_sample_enabled=False)))
    sql_tools = [t.name for t in registry.all() if t.category == "sql"]
    assert sql_tools == ["sql_schema", "sql_query"]
    assert not any("import" in t.name for t in registry.all())


# ----------------------------------------------------------- tools/context
def test_tools_query_imports_through_trusted_session_context(service):
    sid, report = _import(service)
    schema_tool, query_tool = make_sql_tools(service)
    ctx = ToolContext(data_session_id=sid, sql_database=IMPORTED_NAME)
    schema = schema_tool.func(SqlSchemaArgs(), ctx)
    assert "imported by the user" in schema["results"][0]["snippet"]
    out = query_tool.func(SqlQueryArgs(sql=f"SELECT COUNT(*) FROM {report.table}"), ctx)
    assert out["rows"] == [[6]]
    other = ToolContext(data_session_id=uuid.uuid4().hex, sql_database=IMPORTED_NAME)
    with pytest.raises(ToolError):
        query_tool.func(SqlQueryArgs(sql=f"SELECT COUNT(*) FROM {report.table}"), other)


def test_selected_database_restricts_the_model(service):
    sid, report = _import(service)
    _, query_tool = make_sql_tools(service)
    imports_only = ToolContext(data_session_id=sid, sql_database=IMPORTED_NAME)
    with pytest.raises(ToolError, match="Only the selected database"):
        query_tool.func(SqlQueryArgs(database=SAMPLE_NAME, sql="SELECT * FROM orders"),
                        imports_only)
    sample_only = ToolContext(data_session_id=sid, sql_database=SAMPLE_NAME)
    with pytest.raises(ToolError, match="not available"):
        query_tool.func(SqlQueryArgs(sql=f"SELECT * FROM {report.table}"), sample_only)


def test_sample_database_still_works_and_is_labelled(service):
    sid, _ = _import(service)
    sample = service.get(SAMPLE_NAME, sid)
    assert sample.is_sample and "SAMPLE DATA" in sample.description
    assert service.query(SAMPLE_NAME, "SELECT COUNT(*) FROM orders", sid)["rows"] == [[600]]
    prompt = service.schema_for_prompt(sid)
    assert {(d["database"], d["sample_data"], d["imported"]) for d in prompt} == {
        (SAMPLE_NAME, True, False), (IMPORTED_NAME, False, True)}
    assert [d["database"] for d in service.schema_for_prompt(sid, only=IMPORTED_NAME)] == [
        IMPORTED_NAME]
