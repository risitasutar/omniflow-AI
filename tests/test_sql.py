"""Read-only SQLite agent: validation, database-level enforcement, limits, immutability."""

import hashlib
import sqlite3

import pytest

from omniflow.core.config import Settings
from omniflow.sql import sample_data
from omniflow.sql.database import SqlExecutionError, run_query
from omniflow.sql.guard import SqlRejected, validate_sql
from omniflow.sql.service import SAMPLE_NAME, SqlService
from omniflow.tools.registry import ToolContext, ToolError
from omniflow.tools.sql_tools import SqlQueryArgs, SqlSchemaArgs, make_sql_tools

ALLOWED = set(sample_data.ALLOWED_TABLES)


@pytest.fixture
def db_path(tmp_path):
    return sample_data.create_sample_database(tmp_path / "shop.db")


@pytest.fixture
def service(db_path):
    return SqlService(Settings(_env_file=None, sql_sample_path=db_path))


def _fingerprint(path) -> tuple:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    conn = sqlite3.connect(path)
    try:
        counts = tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
                       for t in ("customers", "products", "orders", "order_items", "staff_payroll"))
        tables = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master ORDER BY 1"))
    finally:
        conn.close()
    return digest, counts, tables


# ------------------------------------------------------------------- schema
def test_schema_lists_only_allowlisted_tables_and_marks_restricted_columns(service):
    schema = {t["table"]: t for t in service.schema(SAMPLE_NAME)}
    assert set(schema) == ALLOWED
    assert "staff_payroll" not in schema
    email = next(c for c in schema["customers"]["columns"] if c["name"] == "email")
    assert email["blocked"] is True
    assert schema["products"]["rows"] == 12


# ----------------------------------------------------------- valid queries
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY n DESC",
        "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN products p ON p.product_id = oi.product_id GROUP BY 1",
        "WITH c AS (SELECT city, COUNT(*) AS n FROM customers GROUP BY city) "
        "SELECT * FROM c ORDER BY n DESC LIMIT 3",
        "SELECT strftime('%Y-%m', order_date) AS m, COUNT(*) FROM orders GROUP BY m",
        "SELECT name, RANK() OVER (ORDER BY unit_price DESC) AS r FROM products",
        "SELECT city FROM customers UNION SELECT category FROM products",
        "select * from main.products limit 2;",
    ],
)
def test_valid_read_only_queries_run(service, sql):
    out = service.query(SAMPLE_NAME, sql)
    assert out["rows"] and out["columns"]
    assert out["sample_data"] is True


def test_known_values_from_the_sample_database(service):
    out = service.query(SAMPLE_NAME, "SELECT COUNT(*) FROM orders")
    assert out["rows"] == [[600]]
    out = service.query(SAMPLE_NAME, "SELECT COUNT(DISTINCT category) FROM products")
    assert out["rows"] == [[4]]


def test_empty_result(service):
    out = service.query(SAMPLE_NAME, "SELECT * FROM products WHERE unit_price < 0")
    assert out["rows"] == [] and out["row_count"] == 0


# --------------------------------------------------------------- rejections
@pytest.mark.parametrize(
    "sql,message",
    [
        ("", "empty"),
        ("SELEC name FRM products", "not valid SQL"),
        ("SELECT 1 FROM products; DROP TABLE products", "Exactly one"),
        ("SELECT 1 FROM products; SELECT 2 FROM orders", "Exactly one"),
        ("DELETE FROM orders", "DELETE"),
        ("UPDATE products SET unit_price = 0", "UPDATE"),
        ("INSERT INTO products VALUES (99, 'x', 'y', 1)", "INSERT"),
        ("DROP TABLE orders", "DROP"),
        ("ALTER TABLE orders ADD COLUMN x", "ALTER|COMMAND"),
        ("CREATE TABLE evil (x)", "CREATE"),
        ("REPLACE INTO products VALUES (1, 'x', 'y', 1)", "COMMAND"),
        ("VACUUM INTO 'copy.db'", "COMMAND"),
        ("ATTACH DATABASE 'other.db' AS other", "ATTACH"),
        ("DETACH DATABASE other", "DETACH"),
        ("PRAGMA table_info(orders)", "PRAGMA"),
        ("SELECT load_extension('evil.dll') FROM orders", "load_extension"),
        ("SELECT readfile('C:/Windows/win.ini') FROM orders", "readfile"),
        ("SELECT * FROM staff_payroll", "not available"),
        ("SELECT name FROM sqlite_master", "not available"),
        ("SELECT * FROM temp.secrets", "Schema 'temp'"),
        ("SELECT * FROM other.orders", "Schema 'other'"),
        ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x FROM c",
         "Recursive"),
        ("WITH d AS (SELECT * FROM staff_payroll) SELECT * FROM d", "not available"),
        ("SELECT * FROM products WHERE product_id IN (SELECT staff_id FROM staff_payroll)",
         "not available"),
        ("SELECT email FROM customers", "restricted"),
        ("SELECT 1", "at least one allowed table"),
    ],
)
def test_unsafe_or_invalid_queries_are_blocked_before_execution(service, sql, message):
    with pytest.raises(SqlRejected, match=message):
        service.query(SAMPLE_NAME, sql)


def test_comment_tricks_cannot_smuggle_a_second_statement(service):
    out = service.query(SAMPLE_NAME, "SELECT COUNT(*) FROM orders -- ; DROP TABLE orders")
    assert out["rows"] == [[600]]
    with pytest.raises(SqlRejected):
        service.query(SAMPLE_NAME, "SELECT 1 FROM orders /* x */; DROP TABLE orders")


def test_restricted_column_reads_as_null_via_star(service):
    out = service.query(SAMPLE_NAME, "SELECT * FROM customers LIMIT 3")
    email = out["columns"].index("email")
    assert all(row[email] is None for row in out["rows"])


def test_functions_outside_the_sqlite_allowlist_are_denied_by_sqlite(service):
    with pytest.raises(SqlExecutionError, match="not allowed"):
        service.query(SAMPLE_NAME, "SELECT random() FROM orders")


# ---------------------------------------------- layer 2 alone (no parser)
@pytest.mark.parametrize(
    "raw_sql",
    [
        "DELETE FROM orders",
        "UPDATE products SET unit_price = 0",
        "DROP TABLE orders",
        "INSERT INTO products VALUES (99, 'x', 'y', 1)",
        "SELECT * FROM staff_payroll",
        "SELECT name FROM sqlite_master",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA writable_schema = 1",
        "SELECT load_extension('evil')",
        "CREATE TABLE evil (x)",
    ],
)
def test_database_layer_blocks_attacks_even_without_the_parser(db_path, raw_sql):
    before = _fingerprint(db_path)
    with pytest.raises(SqlExecutionError):
        run_query(db_path, raw_sql, allowed_tables=ALLOWED,
                  blocked_columns=set(sample_data.BLOCKED_COLUMNS), max_rows=10,
                  timeout_seconds=2)
    assert _fingerprint(db_path) == before


def test_database_is_unchanged_after_every_blocked_query(service, db_path):
    before = _fingerprint(db_path)
    for sql in ["DELETE FROM orders", "DROP TABLE orders", "UPDATE products SET name='x'",
                "ATTACH 'x.db' AS y", "PRAGMA journal_mode=DELETE",
                "SELECT 1 FROM orders; DELETE FROM orders", "CREATE TABLE t(x)"]:
        with pytest.raises((SqlRejected, SqlExecutionError)):
            service.query(SAMPLE_NAME, sql)
    assert _fingerprint(db_path) == before


# -------------------------------------------------------------- limits
def test_row_limit(db_path):
    service = SqlService(Settings(_env_file=None, sql_sample_path=db_path, sql_max_rows=5))
    out = service.query(SAMPLE_NAME, "SELECT * FROM orders")
    assert out["row_count"] == 5 and out["truncated"] is True


def test_timeout_stops_expensive_queries(db_path):
    service = SqlService(Settings(_env_file=None, sql_sample_path=db_path,
                                  sql_timeout_seconds=0.2))
    with pytest.raises(SqlExecutionError, match="took longer"):
        service.query(SAMPLE_NAME, "SELECT COUNT(*) FROM order_items a, order_items b, "
                      "order_items c")


def test_query_length_limit(db_path):
    service = SqlService(Settings(_env_file=None, sql_sample_path=db_path,
                                  sql_max_query_chars=100))
    with pytest.raises(SqlRejected, match="longer than"):
        long_sql = "SELECT * FROM orders WHERE " + " OR ".join(["1=1"] * 50)  # noqa: S608
        service.query(SAMPLE_NAME, long_sql)


def test_unknown_column_is_reported_safely(service, db_path):
    with pytest.raises(SqlExecutionError) as err:
        service.query(SAMPLE_NAME, "SELECT nonexistent FROM orders")
    assert str(db_path) not in str(err.value)


# ------------------------------------------------- configuration and paths
def test_user_database_needs_an_explicit_table_allowlist(tmp_path):
    user_db = sample_data.create_sample_database(tmp_path / "mine.sqlite")
    s = Settings(_env_file=None, sql_sample_enabled=False, sql_database_path=user_db)
    service = SqlService(s)
    assert service.databases() == []
    problem = service.databases(include_unavailable=True)[0].problem
    assert "OMNIFLOW_SQL_ALLOWED_TABLES" in problem and str(tmp_path) not in problem

    s = Settings(_env_file=None, sql_sample_enabled=False, sql_database_path=user_db,
                 sql_allowed_tables="products, orders", sql_database_name="shop")
    service = SqlService(s)
    assert [d.name for d in service.databases()] == ["shop"]
    with pytest.raises(SqlRejected, match="not available"):
        service.query("shop", "SELECT * FROM customers")


@pytest.mark.parametrize("name", ["missing.db", "notes.txt"])
def test_bad_user_database_paths_are_unavailable(tmp_path, name):
    s = Settings(_env_file=None, sql_sample_enabled=False, sql_database_path=tmp_path / name,
                 sql_allowed_tables="t")
    service = SqlService(s)
    assert service.databases() == []


def test_database_names_not_paths_select_the_database(service):
    for name in ["../shop.db", "C:/Windows/system.db", "sample_shop.db", ":memory:"]:
        with pytest.raises(SqlRejected, match="Unknown database"):
            service.query(name, "SELECT * FROM products")


# ------------------------------------------------------------------- tools
def test_sql_tools_return_structured_results_and_safe_errors(service):
    schema_tool, query_tool = make_sql_tools(service)
    schema = schema_tool.func(SqlSchemaArgs(), ToolContext())
    assert "SAMPLE DATA" in schema["results"][0]["snippet"]
    out = query_tool.func(SqlQueryArgs(sql="SELECT COUNT(*) AS n FROM products"), ToolContext())
    assert out["rows"] == [[12]] and out["source_type"] == "sql"
    assert "Executed SQL" in out["results"][0]["snippet"]
    with pytest.raises(ToolError) as err:
        query_tool.func(SqlQueryArgs(sql="DROP TABLE products"), ToolContext())
    assert err.value.category == "sql_rejected" and not err.value.retryable


def test_validate_sql_returns_normalised_sql_and_tables():
    validated = validate_sql("select  name from products", ALLOWED, max_chars=1000)
    assert validated.sql == "SELECT name FROM products"
    assert validated.tables == ("products",)
