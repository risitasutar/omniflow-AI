"""Synthetic tabular data for tests (built in memory; no files in the repository)."""

from __future__ import annotations

import io

import pandas as pd

# Small table whose answers are easy to check by hand.
SMALL_CSV = (
    b"order_id,region,product,units,unit_price,order_date\n"
    b"1,North,Widget,10,2.5,2025-01-05\n"
    b"2,South,Widget,4,2.5,2025-01-20\n"
    b"3,North,Gadget,1,100,2025-02-03\n"
    b"4,East,Gadget,2,100,2025-02-14\n"
    b"5,South,Doohickey,5,8,2025-03-01\n"
    b"6,North,Widget,,2.5,2025-03-09\n"
)
# units: 10, 4, 1, 2, 5, missing -> sum 22, mean 4.4
# revenue by product (units * unit_price): Widget 35, Gadget 300, Doohickey 40


def sales_frame(rows: int = 120) -> pd.DataFrame:
    """A larger deterministic sales table with one clear outlier and one duplicate row."""
    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    records = []
    for i in range(rows):
        units = (i * 7) % 13 + 1
        price = [10.0, 25.0, 40.0, 5.0, 60.0][i % 5]
        records.append({
            "order_id": 1000 + i,
            "order_date": f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "region": regions[i % 4],
            "product": products[i % 5],
            "units": units,
            "unit_price": price,
            "revenue": units * price,
        })
    records[7]["revenue"] = 99_999.0  # an obvious outlier
    records.append(dict(records[3]))  # one exact duplicate row
    return pd.DataFrame(records)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
