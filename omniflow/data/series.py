"""Turn a (date column, numeric column) pair into a validated, regular time series.

Rules (all reported back to the user):
- The date column must be a datetime column or parse as dates for at least 90% of rows;
  unparseable dates are dropped and counted. The target must be a numeric, non-identifier
  column; identifiers (ids, codes, phone numbers...) are never treated as quantities.
- Rows with a missing target are dropped and counted (never imputed).
- Several rows in one period (including duplicate timestamps) are combined with the chosen
  aggregation: sum (totals such as revenue or units) or mean (levels such as prices or
  temperatures). The choice is the user's and is stated in every result.
- The frequency is the user's choice or is inferred from the median spacing of timestamps;
  the share of irregular spacings is reported.
- Empty periods are gaps. They are never filled silently: trend analysis shows them as gaps,
  and forecasting refuses gappy series unless the user explicitly chooses "zero" (only
  meaningful for sums/counts) or "interpolate" (at most 10% of periods, runs of at most 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from omniflow.data.analysis import AnalysisError, resolve_column

Frequency = Literal["auto", "day", "week", "month", "quarter", "year"]
Aggregation = Literal["sum", "mean"]
GapPolicy = Literal["refuse", "zero", "interpolate"]
FREQ_CODES = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
SEASON_LENGTH = {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": None}
_SPACING_DAYS = [("day", 1), ("week", 7), ("month", 30.44), ("quarter", 91.3), ("year", 365.25)]
MIN_PARSE_RATE = 0.9


@dataclass
class Series:
    values: np.ndarray  # float, may contain NaN for gaps
    index: pd.DatetimeIndex
    frequency: str
    aggregation: str
    date_column: str
    value_column: str
    report: dict = field(default_factory=dict)
    counts: np.ndarray | None = None  # rows combined into each period

    @property
    def n(self) -> int:
        return int(len(self.values))

    @property
    def gaps(self) -> int:
        return int(np.isnan(self.values).sum())

    @property
    def season_length(self) -> int | None:
        return SEASON_LENGTH.get(self.frequency)

    def labels(self) -> list[str]:
        fmt = {"day": "%Y-%m-%d", "week": "%Y-%m-%d", "month": "%Y-%m", "quarter": "%Y-%m",
               "year": "%Y"}[self.frequency]
        if self.frequency == "quarter":
            return [f"{t.year}-Q{(t.month - 1) // 3 + 1}" for t in self.index]
        return [t.strftime(fmt) for t in self.index]


def infer_frequency(dates: pd.Series) -> tuple[str, float]:
    """(frequency, share of spacings that don't match it)."""
    unique = pd.Series(sorted(dates.dropna().unique()))
    if len(unique) < 3:
        raise AnalysisError("At least 3 distinct dates are needed to tell the frequency.")
    spacing = unique.diff().dropna().dt.total_seconds() / 86400
    median = float(spacing.median())
    name = min(_SPACING_DAYS, key=lambda item: abs(np.log(max(median, 1e-6) / item[1])))[0]
    # Irregular = a gap between consecutive dates more than 50% away from the typical gap.
    irregular = float(((spacing - median).abs() > median * 0.5).mean())
    return name, irregular


def median_spacing_days(dates: pd.Series) -> float:
    unique = pd.Series(sorted(dates.dropna().unique()))
    return float((unique.diff().dropna().dt.total_seconds() / 86400).median())


def _stable_frequency(values: pd.Series, start: str) -> str:
    """Finest frequency (from `start` upward) whose row counts per period are perfectly
    regular or average at least 5 rows, so sums aren't driven by how dates fall into bins."""
    names = [name for name, _ in _SPACING_DAYS]
    for name in names[names.index(start):]:
        counts = values.resample(FREQ_CODES[name]).count()
        inner = counts.iloc[1:-1] if len(counts) > 4 else counts
        inner = inner[inner > 0]
        if len(inner) == 0:
            continue
        if inner.std(ddof=0) == 0 or inner.mean() >= 5:
            if len(counts) >= 3 or name == start:
                return name
    return start


def prepare(frame: pd.DataFrame, roles: dict[str, str], *, date_column: str, value_column: str,
            aggregation: Aggregation = "sum", frequency: Frequency = "auto",
            max_periods: int = 5000) -> Series:
    if not date_column or not value_column:
        raise AnalysisError("Choose both a date column and a numeric column.")
    date = resolve_column(frame, date_column)
    value = resolve_column(frame, value_column)
    if date == value:
        raise AnalysisError("The date column and the value column must be different.")
    role = roles.get(value)
    if role == "identifier":
        raise AnalysisError(f"'{value}' looks like an identifier (an id or code), not a "
                            "quantity, so it can't be analysed as a time series.")
    if role not in ("numeric",) and not pd.api.types.is_numeric_dtype(frame[value]):
        raise AnalysisError(f"'{value}' is not numeric. Choose a numeric column.")
    raw_dates = frame[date]
    if pd.api.types.is_datetime64_any_dtype(raw_dates):
        dates = raw_dates
    else:
        dates = pd.to_datetime(raw_dates.astype("string"), errors="coerce", format="mixed")
    present = raw_dates.notna()
    parsed = dates.notna() & present
    if not present.any() or parsed.sum() / max(1, present.sum()) < MIN_PARSE_RATE:
        raise AnalysisError(f"'{date}' doesn't look like a date column (fewer than "
                            f"{MIN_PARSE_RATE:.0%} of its values are dates).")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    values = pd.to_numeric(frame[value], errors="coerce")
    usable = parsed & values.notna()
    report = {
        "rows_in_dataset": int(len(frame)), "rows_used": int(usable.sum()),
        "unparseable_dates": int((present & ~parsed).sum()),
        "missing_dates": int((~present).sum()),
        "missing_values_dropped": int((parsed & values.isna()).sum()),
        "aggregation": aggregation,
    }
    if report["rows_used"] < 3:
        raise AnalysisError("Fewer than 3 rows have both a valid date and a value.")
    data = pd.DataFrame({"date": dates[usable], "value": values[usable].astype(float)})
    report["duplicate_timestamps"] = int(data["date"].duplicated().sum())
    indexed = data.set_index("date")["value"]
    if frequency == "auto":
        frequency, irregular = infer_frequency(data["date"])
        frequency = _stable_frequency(indexed, frequency)
        report["frequency_source"] = ("inferred from the spacing of dates (the finest period "
                                      "with regular or well-filled periods)")
    else:
        _, irregular = infer_frequency(data["date"])
        report["frequency_source"] = "chosen by the user"
    report["irregular_spacing_share"] = round(irregular, 3)
    grouped = indexed.resample(FREQ_CODES[frequency])
    counts = grouped.count()
    filled = counts[counts > 0]
    report["rows_per_period"] = {"min": int(filled.min()), "median": float(filled.median()),
                                 "max": int(filled.max())}
    if aggregation == "sum" and filled.max() > filled.min() and \
            filled.std() / filled.mean() > 0.2:
        report["count_note"] = ("Periods contain different numbers of rows "
                                f"({int(filled.min())}-{int(filled.max())}), so sums partly "
                                "reflect how many records each period has.")
    series = grouped.sum(min_count=1) if aggregation == "sum" else grouped.mean()
    series[counts == 0] = np.nan  # an empty period is a gap, never a silent zero
    # A first/last period the data only partly covers would look like a fake drop or jump.
    spacing = pd.Timedelta(days=1.5 * median_spacing_days(data["date"]))
    period_ends = list(series.index[1:]) + [series.index[-1] + series.index.freq]
    trimmed = []
    if len(series) > 3 and data["date"].min() - series.index[0] > spacing:
        trimmed.append(str(series.index[0].date()))
        series, counts = series.iloc[1:], counts.iloc[1:]
        period_ends = period_ends[1:]
    if len(series) > 3 and period_ends[-1] - pd.Timedelta(days=1) - data["date"].max() > spacing:
        trimmed.append(str(series.index[-1].date()))
        series, counts = series.iloc[:-1], counts.iloc[:-1]
    if trimmed:
        report["partial_periods_dropped"] = trimmed
        report["partial_note"] = ("Dropped partial first/last period(s) the data only partly "
                                  f"covers: {', '.join(trimmed)}.")
    if len(series) > max_periods:
        raise AnalysisError(f"The series has {len(series):,} {frequency} periods; the limit is "
                            f"{max_periods:,}. Choose a coarser frequency.")
    report["periods"] = int(len(series))
    report["rows_per_period_max"] = int(counts.max())
    gaps = series.isna()
    report["empty_periods"] = int(gaps.sum())
    runs = (gaps != gaps.shift()).cumsum()[gaps]
    report["longest_gap"] = int(runs.value_counts().max()) if gaps.any() else 0
    report["first_period"] = str(series.index[0].date())
    report["last_period"] = str(series.index[-1].date())
    if report["duplicate_timestamps"] or report["rows_per_period_max"] > 1:
        report["combining_rule"] = (f"Rows in the same {frequency} were combined with "
                                    f"{aggregation}.")
    return Series(series.to_numpy(dtype=float), series.index, frequency, aggregation, date,
                  value, report, counts.to_numpy(dtype=float))


def fill_gaps(series: Series, policy: GapPolicy) -> Series:
    """Apply the user's explicit gap policy for forecasting (never applied silently)."""
    if series.gaps == 0:
        return series
    if policy == "refuse":
        raise AnalysisError(
            f"The series has {series.gaps} empty {series.frequency}(s) (longest run "
            f"{series.report['longest_gap']}). Forecasting needs a complete series: choose a "
            "coarser frequency, or explicitly treat empty periods as zero (only for totals) "
            "or interpolate short gaps.")
    values = series.values.copy()
    if policy == "zero":
        if series.aggregation != "sum":
            raise AnalysisError("Treating empty periods as zero only makes sense for totals "
                                "(sum), not averages.")
        values[np.isnan(values)] = 0.0
        note = f"{series.gaps} empty period(s) treated as zero, as you chose."
    else:
        if series.gaps / series.n > 0.10 or series.report["longest_gap"] > 2:
            raise AnalysisError("Too many or too long gaps to interpolate (limit: 10% of "
                                "periods and runs of at most 2).")
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
        note = f"{series.gaps} empty period(s) filled by linear interpolation, as you chose."
    report = dict(series.report) | {"gap_policy": policy, "gap_note": note}
    return Series(values, series.index, series.frequency, series.aggregation,
                  series.date_column, series.value_column, report, series.counts)
