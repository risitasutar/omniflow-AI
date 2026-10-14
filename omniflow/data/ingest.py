"""Load CSV / XLSX uploads into DataFrames, safely and within limits.

Everything happens in memory. Uploaded names are display labels only (never paths).
XLSX files are opened with openpyxl (which uses defusedxml), formulas are read as their
cached values and never evaluated, and macro-enabled files are not accepted.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from omniflow.documents.pdf import safe_display_name

XLSX_MAGIC = b"PK\x03\x04"
MAX_UNZIPPED_BYTES = 250_000_000  # zip-bomb guard for .xlsx
_DATE_HINT = re.compile(r"(date|time|day|month|year|timestamp|created|updated|_at$)", re.I)


class DataError(Exception):
    """A data problem with a safe, user-facing message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass
class LoadedTable:
    name: str  # display label, e.g. "sales.xlsx [Q1]"
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Limits:
    max_bytes: int
    max_rows: int
    max_columns: int
    max_memory_mb: float
    max_sheets: int


def validate_upload(name: str | None, data: bytes, max_bytes: int) -> str:
    lower = (name or "").lower()
    if lower.endswith(".csv"):
        kind = "csv"
    elif lower.endswith(".xlsx"):
        kind = "xlsx"
    else:
        raise DataError("unsupported_type", "Only .csv and .xlsx files are supported "
                        "(macro-enabled .xlsm and old .xls files are not).")
    if not data:
        raise DataError("empty_file", "The file is empty.")
    if len(data) > max_bytes:
        raise DataError("too_large", f"The file is larger than the {max_bytes / 1e6:g} MB limit.")
    if kind == "xlsx" and not data.startswith(XLSX_MAGIC):
        raise DataError("unsupported_type", "The file is not a valid .xlsx workbook.")
    if kind == "csv" and b"\x00" in data[:8192]:
        raise DataError("unsupported_type", "The file looks binary, not like a text CSV.")
    return kind


def _decode(data: bytes) -> tuple[str, str | None]:
    for encoding in ("utf-8-sig", "utf-16"):
        if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        try:
            return data.decode(encoding), None
        except UnicodeDecodeError:
            pass
    # Single-byte fallback never fails; say so, since some characters may be wrong.
    return data.decode("cp1252", errors="replace"), (
        "The file is not UTF-8; it was read as Windows-1252, so a few characters may be wrong."
    )


def _sniff_delimiter(text: str) -> str:
    sample = text[:20_000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def clean_columns(frame: pd.DataFrame) -> list[str]:
    """Make column names non-empty, trimmed, unique strings. Returns warnings."""
    warnings: list[str] = []
    seen: dict[str, int] = {}
    names: list[str] = []
    renamed_dupes = []
    for i, raw in enumerate(frame.columns):
        name = re.sub(r"\s+", " ", str(raw)).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        base = name
        if base.lower() in seen:
            seen[base.lower()] += 1
            name = f"{base}_{seen[base.lower()]}"
            renamed_dupes.append(base)
        else:
            seen[base.lower()] = 1
        names.append(name[:100])
    frame.columns = names
    if renamed_dupes:
        warnings.append("Duplicate column names were renamed with a numeric suffix: "
                        + ", ".join(sorted(set(renamed_dupes)))[:300])
    return warnings


def coerce_types(frame: pd.DataFrame) -> list[str]:
    """Convert text columns that are mostly numbers or dates. Returns warnings."""
    warnings: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        present = series.dropna()
        if present.empty:
            continue
        text = present.astype(str).str.strip()
        numeric = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
        if numeric.notna().mean() >= 0.95:
            bad = int(numeric.isna().sum())
            frame[column] = pd.to_numeric(
                series.astype(str).str.strip().str.replace(",", "", regex=False),
                errors="coerce",
            ).where(series.notna())
            if bad:
                warnings.append(f"'{column}': {bad} non-numeric value(s) treated as missing.")
            continue
        if _DATE_HINT.search(str(column)) or text.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
            dates = pd.to_datetime(text, errors="coerce", format="mixed")
            if dates.notna().mean() >= 0.9:
                bad = int(dates.isna().sum())
                frame[column] = pd.to_datetime(series, errors="coerce", format="mixed")
                if bad:
                    warnings.append(f"'{column}': {bad} unparseable date(s) treated as missing.")
    return warnings


def _check_limits(frame: pd.DataFrame, limits: Limits, label: str) -> None:
    rows, cols = frame.shape
    if rows > limits.max_rows:
        raise DataError("too_many_rows", f"{label} has more than {limits.max_rows:,} rows.")
    if cols > limits.max_columns:
        raise DataError("too_many_columns",
                        f"{label} has {cols} columns; the limit is {limits.max_columns}.")
    memory_mb = frame.memory_usage(deep=True).sum() / 1e6
    if memory_mb > limits.max_memory_mb:
        raise DataError("too_much_memory",
                        f"{label} needs about {memory_mb:.0f} MB of memory; the limit is "
                        f"{limits.max_memory_mb:g} MB.")


def _promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the first row as the header (read with header=None so duplicates survive)."""
    if frame.empty:
        return frame
    header = ["" if pd.isna(v) else v for v in frame.iloc[0].tolist()]
    frame = frame.iloc[1:].copy()
    frame.columns = header
    return frame


def _finish(frame: pd.DataFrame, label: str, limits: Limits, warnings: list[str]) -> LoadedTable:
    frame = _promote_header(frame.dropna(how="all"))
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    if frame.empty or frame.shape[1] == 0:
        raise DataError("empty_table", f"{label} contains no data.")
    warnings += clean_columns(frame)
    _check_limits(frame, limits, label)
    warnings += coerce_types(frame)
    frame = frame.reset_index(drop=True)
    return LoadedTable(label, frame, warnings)


def load_csv(name: str, data: bytes, limits: Limits) -> LoadedTable:
    text, note = _decode(data)
    warnings = [note] if note else []
    delimiter = _sniff_delimiter(text)
    kwargs = dict(sep=delimiter, nrows=limits.max_rows + 2, dtype=object, header=None,
                  keep_default_na=True, skipinitialspace=True)
    try:
        frame = pd.read_csv(io.StringIO(text), **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DataError("empty_table", "The CSV file has no data.") from exc
    except pd.errors.ParserError:
        try:
            frame = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kwargs)
        except (pd.errors.ParserError, ValueError) as exc:
            raise DataError("malformed", "The CSV file could not be parsed.") from exc
        warnings.append("Some malformed lines (wrong number of fields) were skipped.")
    except (ValueError, UnicodeError) as exc:
        raise DataError("malformed", "The CSV file could not be parsed.") from exc
    return _finish(frame, name, limits, warnings)


def load_xlsx(name: str, data: bytes, limits: Limits) -> list[LoadedTable]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > MAX_UNZIPPED_BYTES:
                raise DataError("too_large", "The workbook expands to too much data to open.")
            if any(info.filename.lower().endswith("vbaproject.bin") for info in archive.infolist()):
                raise DataError("unsupported_type", "Workbooks containing macros are not accepted.")
    except zipfile.BadZipFile as exc:
        raise DataError("malformed", "The .xlsx file is damaged or not a workbook.") from exc
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl",
                               dtype=object, nrows=limits.max_rows + 2, header=None)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises many types on bad input
        raise DataError("malformed", "The .xlsx file could not be read.") from exc
    tables: list[LoadedTable] = []
    skipped = []
    for sheet_name, frame in sheets.items():
        label = f"{name} [{str(sheet_name)[:40]}]" if len(sheets) > 1 else name
        if len(tables) >= limits.max_sheets:
            skipped.append(str(sheet_name))
            continue
        try:
            tables.append(_finish(frame, label, limits, []))
        except DataError as exc:
            if exc.code != "empty_table":
                raise
    if not tables:
        raise DataError("empty_table", "The workbook contains no data.")
    if skipped:
        tables[0].warnings.append(f"Only the first {limits.max_sheets} sheets were loaded.")
    return tables


def load_upload(name: str | None, data: bytes, limits: Limits) -> list[LoadedTable]:
    kind = validate_upload(name, data, limits.max_bytes)
    label = safe_display_name(name)
    if kind == "csv":
        return [load_csv(label, data, limits)]
    return load_xlsx(label, data, limits)


def json_safe(value):
    """Convert numpy / pandas scalars to plain JSON-friendly Python values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (int, bool, str)):
        return value if not isinstance(value, str) else value[:300]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)[:300]
