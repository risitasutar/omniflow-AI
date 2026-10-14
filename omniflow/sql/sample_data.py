"""Create the synthetic SAMPLE database (fictional shop data - not real people or sales).

    python -m omniflow.sql.sample_data          # create if missing
    python -m omniflow.sql.sample_data --force  # recreate

The `staff_payroll` table is deliberately NOT on the allowlist and `customers.email` is a
blocked column, so the access controls can be demonstrated safely.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ALLOWED_TABLES = {"customers", "products", "orders", "order_items"}
BLOCKED_COLUMNS = {("customers", "email")}

_CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"]
_SEGMENTS = ["Consumer", "Corporate", "Small Business"]
_PRODUCTS = [
    ("Wireless Mouse", "Electronics", 799.0), ("Mechanical Keyboard", "Electronics", 3499.0),
    ("USB-C Hub", "Electronics", 1899.0), ("27in Monitor", "Electronics", 14999.0),
    ("Office Chair", "Furniture", 8999.0), ("Standing Desk", "Furniture", 21999.0),
    ("Desk Lamp", "Furniture", 1299.0), ("Notebook A5", "Stationery", 149.0),
    ("Gel Pens (10)", "Stationery", 249.0), ("Planner 2026", "Stationery", 599.0),
    ("Coffee Beans 1kg", "Pantry", 1199.0), ("Green Tea (100)", "Pantry", 449.0),
]


def create_sample_database(path: Path, *, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    rng = random.Random(20261011)  # noqa: S311 - synthetic sample data, not security
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript("""
            CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                city TEXT, segment TEXT, signup_date TEXT, email TEXT);
            CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                category TEXT, unit_price REAL);
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER
                REFERENCES customers(customer_id), order_date TEXT, status TEXT);
            CREATE TABLE order_items (order_id INTEGER REFERENCES orders(order_id),
                product_id INTEGER REFERENCES products(product_id), quantity INTEGER,
                unit_price REAL);
            CREATE TABLE staff_payroll (staff_id INTEGER PRIMARY KEY, name TEXT,
                salary REAL, bank_account TEXT);
        """)
        start = date(2025, 1, 1)
        customers = [(i, f"Customer {i:03d}", rng.choice(_CITIES), rng.choice(_SEGMENTS),
                      (start + timedelta(days=rng.randint(0, 300))).isoformat(),
                      f"customer{i:03d}@example.invalid") for i in range(1, 121)]
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)",
                         [(i, *p) for i, p in enumerate(_PRODUCTS, start=1)])
        order_id = 0
        for _ in range(600):
            order_id += 1
            day = start + timedelta(days=rng.randint(0, 545))
            status = rng.choices(["delivered", "shipped", "cancelled", "returned"],
                                 [80, 8, 8, 4])[0]
            conn.execute("INSERT INTO orders VALUES (?,?,?,?)",
                         (order_id, rng.randint(1, 120), day.isoformat(), status))
            for product_id in rng.sample(range(1, len(_PRODUCTS) + 1), rng.randint(1, 3)):
                price = _PRODUCTS[product_id - 1][2]
                conn.execute("INSERT INTO order_items VALUES (?,?,?,?)",
                             (order_id, product_id, rng.randint(1, 4), price))
        conn.executemany("INSERT INTO staff_payroll VALUES (?,?,?,?)",
                         [(i, f"Staff {i}", 50000.0 + i * 1000, f"XX00-SAMPLE-{i:04d}")
                          for i in range(1, 11)])
        conn.commit()
    finally:
        conn.close()
    tmp.replace(path)
    return path


if __name__ == "__main__":
    from omniflow.core.config import get_settings

    parser = argparse.ArgumentParser(description="Create the synthetic sample database")
    parser.add_argument("--force", action="store_true", help="recreate it")
    result = create_sample_database(get_settings().sql_sample_path,
                                    force=parser.parse_args().force)
    print(f"Sample database ready: {result}")
