"""`weather_lookup`: current conditions and daily forecasts from Open-Meteo (keyless).

Two GET requests per lookup, both through the locked-down `safe_get` (HTTPS only, fixed
host allowlist, public-address check, size cap, timeout):

1. https://geocoding-api.open-meteo.com/v1/search - turns the place name into coordinates.
   OmniFlow never guesses coordinates. If several different places match and none clearly
   dominates, the tool asks the user to choose instead of picking one.
2. https://api.open-meteo.com/v1/forecast - current conditions and a 1-16 day daily
   forecast for those coordinates, with `timezone=auto`, so dates and times are local.

Open-Meteo's free API needs no key for non-commercial use (fewer than 10,000 calls/day,
5,000/hour, 600/minute) and its data are licensed CC BY 4.0, credited as "Weather data by
Open-Meteo.com" (open-meteo.com terms and licence pages, checked 2026-10-14).

"Current" values are model estimates for the location, not station observations, and
forecasts are predictions. Missing values are reported as "not provided", never filled in.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniflow.core.config import Settings
from omniflow.research.http import FetchError, safe_get
from omniflow.tools.registry import (
    Availability,
    AvailabilityStatus,
    Permission,
    SideEffect,
    ToolContext,
    ToolError,
    ToolSpec,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_HOSTS = frozenset({"geocoding-api.open-meteo.com", "api.open-meteo.com"})
ATTRIBUTION = "Weather data by Open-Meteo.com"
ATTRIBUTION_URL = "https://open-meteo.com/"
TIMEOUT = 10.0  # per HTTP request
MAX_DAYS = 16  # Open-Meteo's forecast_days maximum
GEOCODE_COUNT = 10
MAX_BYTES = 500_000
DOMINANCE = 10  # a match is "clearly the one meant" if 10x more populous than all others

UNIT_PARAMS = {
    "metric": {"temperature_unit": "celsius", "wind_speed_unit": "kmh",
               "precipitation_unit": "mm"},
    "imperial": {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                 "precipitation_unit": "inch"},
}
UNIT_LABELS = {
    "metric": {"temperature": "°C", "wind": "km/h", "precipitation": "mm"},
    "imperial": {"temperature": "°F", "wind": "mph", "precipitation": "in"},
}
CURRENT_VARS = ("weather_code", "temperature_2m", "apparent_temperature",
                "relative_humidity_2m", "precipitation", "wind_speed_10m",
                "wind_direction_10m", "wind_gusts_10m")
DAILY_VARS = ("weather_code", "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
              "precipitation_probability_max", "wind_speed_10m_max")

# WMO weather interpretation codes, as listed in Open-Meteo's documentation.
WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
    48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snowfall", 73: "Moderate snowfall",
    75: "Heavy snowfall", 77: "Snow grains", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LOCATION = re.compile(r"^[\w\s,.'()\-]+$")
_LOCAL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_TIMEZONE = re.compile(r"^[A-Za-z_]+(?:/[A-Za-z0-9_+\-]+){0,2}$|^GMT$|^UTC$")
NOT_PROVIDED = "not provided"


class WeatherArgs(BaseModel):
    """Arguments for weather_lookup."""

    model_config = ConfigDict(extra="forbid")
    location: str = Field(description="Place name, optionally narrowed with region and/or "
                          "country after commas, e.g. 'Springfield, Illinois' or 'Paris, "
                          "France'. 2-100 characters.")
    days: int = Field(default=3, ge=1, le=MAX_DAYS,
                      description=f"Days of daily forecast starting today (1-{MAX_DAYS}).")
    units: Literal["metric", "imperial"] = Field(
        default="metric", description="'metric' (°C, km/h, mm) or 'imperial' (°F, mph, in).")
    country_code: str | None = Field(default=None, description="Optional ISO 3166-1 alpha-2 "
                                     "country code to narrow the place, e.g. 'IN' or 'US'.")

    @field_validator("location")
    @classmethod
    def _clean_location(cls, value: str) -> str:
        cleaned = " ".join(_CONTROL.sub(" ", value).split()).strip(" ,")
        if not 2 <= len(cleaned) <= 100:
            raise ValueError("location must be 2-100 characters")
        if not _LOCATION.match(cleaned):
            raise ValueError("location may contain only letters, digits, spaces and , . ' ( ) -")
        if len(cleaned.split(",")[0].strip()) < 2:
            raise ValueError("the place name before the first comma needs 2+ characters")
        return cleaned

    @field_validator("country_code")
    @classmethod
    def _clean_country(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        code = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{2}", code):
            raise ValueError("country_code must be a 2-letter ISO code such as 'IN'")
        return code


# ------------------------------------------------------------------ network
def _geocode_get(params: dict, timeout: float) -> bytes:
    """The only function that calls the geocoding API (blocked in tests; mocked per test)."""
    body, _ = safe_get(GEOCODE_URL, allowed_hosts=WEATHER_HOSTS, params=params,
                       timeout=timeout, max_bytes=MAX_BYTES)
    return body


def _forecast_get(params: dict, timeout: float) -> bytes:
    """The only function that calls the forecast API (blocked in tests; mocked per test)."""
    body, _ = safe_get(FORECAST_URL, allowed_hosts=WEATHER_HOSTS, params=params,
                       timeout=timeout, max_bytes=MAX_BYTES)
    return body


def _tool_error(exc: FetchError, service: str) -> ToolError:
    where = f"Open-Meteo's {service} service"
    if exc.category == "timeout":
        return ToolError("timeout", f"{where} didn't respond within {TIMEOUT:g} seconds.",
                         retryable=True)
    if exc.category == "network":
        return ToolError("network", f"{where} couldn't be reached (network error).",
                         retryable=True)
    if exc.category == "rate_limited":  # minute/hour/day limits: an instant retry can't help
        return ToolError("rate_limited", f"{where} is rate-limiting requests (HTTP 429). Try "
                         "again later.")
    if exc.status is not None and exc.status >= 500:
        return ToolError("provider_error", f"{where} is temporarily unavailable (HTTP "
                         f"{exc.status}).", retryable=True)
    if exc.status is not None:
        return ToolError("invalid_request", f"{where} rejected the request (HTTP "
                         f"{exc.status}).")
    return ToolError("provider_error", f"{where} could not be used safely ({exc.category}).")


def _fetch_json(fetch, params: dict, service: str) -> Any:
    try:
        body = fetch(params, TIMEOUT)
    except FetchError as exc:
        raise _tool_error(exc, service) from exc
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ToolError("bad_response", f"Open-Meteo's {service} service returned a response "
                        "that isn't valid JSON.") from exc


# ----------------------------------------------------------------- geocoding
def _text(value: Any, limit: int) -> str:
    return " ".join(_CONTROL.sub(" ", str(value or "")).split())[:limit]


def _fold(value: str) -> str:
    """Case- and accent-insensitive form, so 'Zurich' matches 'Zürich'."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def place_label(place: dict) -> str:
    parts = [place["name"], place.get("admin1"), place.get("country")]
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return ", ".join(seen)


def _places(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        raise ToolError("bad_response", "Open-Meteo's geocoding service returned a response in "
                        "an unexpected format.")
    raw = data.get("results")
    if raw is None:
        return []  # Open-Meteo omits "results" when nothing matches
    if not isinstance(raw, list):
        raise ToolError("bad_response", "Open-Meteo's geocoding service returned a response in "
                        "an unexpected format.")
    places = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lat, lon = _finite(item.get("latitude")), _finite(item.get("longitude"))
        name = _text(item.get("name"), 100)
        if not name or lat is None or lon is None or not (-90 <= lat <= 90 and
                                                           -180 <= lon <= 180):
            continue  # unusable without real coordinates
        population = item.get("population")
        code = _text(item.get("country_code"), 2).upper()
        places.append({
            "name": name, "admin1": _text(item.get("admin1"), 100),
            "admin2": _text(item.get("admin2"), 100), "country": _text(item.get("country"), 100),
            "country_code": code if re.fullmatch(r"[A-Z]{2}", code) else "",
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "population": population if isinstance(population, int) and not isinstance(
                population, bool) and population >= 0 else None,
        })
    return places


def resolve_location(data: Any, query: str, country_code: str | None
                     ) -> tuple[dict, list[str]]:
    """Pick the place the user meant, or raise a ToolError asking them to be specific.

    Returns (place, labels of other places with the same name)."""
    name, *qualifiers = [p.strip() for p in query.split(",") if p.strip()]
    places = _places(data)
    if country_code:
        places = [p for p in places if p["country_code"] == country_code]
    exact = [p for p in places if _fold(p["name"]) == _fold(name)]
    places = exact or places  # prefer exact names over prefix matches ("York" vs "Yorkton")
    named = places
    for qualifier in qualifiers:
        q = _fold(qualifier)
        places = [p for p in places if q in {_fold(p["admin1"]), _fold(p["admin2"]),
                                             _fold(p["country"]), _fold(p["country_code"])}]
    unique: dict[tuple, dict] = {}
    for p in places:  # one entry per distinct place label
        unique.setdefault((_fold(p["name"]), _fold(p["admin1"]), p["country_code"]), p)
    candidates = list(unique.values())
    if not candidates:
        if named and qualifiers:
            listed = "; ".join(place_label(p) for p in named[:5])
            raise ToolError("location_not_found", f"I found places named '{name}' but none "
                            f"matching '{', '.join(qualifiers)}': {listed}. Please give the "
                            "region or country as listed.")
        where = f" in country {country_code}" if country_code else ""
        raise ToolError("location_not_found", f"Open-Meteo's geocoding found no place named "
                        f"'{query}'{where}. Check the spelling or add the country, e.g. "
                        "'Paris, France'.")
    if len(candidates) == 1:
        return candidates[0], []
    ranked = sorted(candidates, key=lambda p: p["population"] or 0, reverse=True)
    top, others = ranked[0], ranked[1:]
    runner_up = max(p["population"] or 0 for p in others)
    if top["population"] and top["population"] >= DOMINANCE * max(runner_up, 1):
        return top, [place_label(p) for p in others[:5]]
    listed = "; ".join(place_label(p) for p in ranked[:5])
    more = f" (and {len(ranked) - 5} more)" if len(ranked) > 5 else ""
    raise ToolError("ambiguous_location", f"'{query}' matches several places: {listed}{more}. "
                    f"Which one do you mean? For example: '{place_label(ranked[0])}'.")


# ------------------------------------------------------------------ forecast
def _value(raw: Any) -> float | None:
    number = _finite(raw)
    return None if number is None else round(number, 1)


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return NOT_PROVIDED
    text = f"{value:g}"
    return f"{text} {unit}".strip() if unit not in ("%", "°") else f"{text}{unit}"


def _condition(code: float | None) -> str:
    if code is None:
        return NOT_PROVIDED
    return WMO.get(int(code), f"unknown condition (WMO code {int(code)})")


def parse_forecast(data: Any, days: int, units: str) -> dict:
    """Validate Open-Meteo's forecast response into plain, bounded values (None = missing)."""
    if not isinstance(data, dict):
        raise ToolError("bad_response", "Open-Meteo's forecast service returned a response in "
                        "an unexpected format.")
    current, daily = data.get("current"), data.get("daily")
    if not isinstance(current, dict) and not isinstance(daily, dict):
        raise ToolError("bad_response", "Open-Meteo's forecast response contained neither "
                        "current conditions nor a daily forecast.")
    tz = _text(data.get("timezone"), 64)
    offset = data.get("utc_offset_seconds")
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) and abs(
        offset) <= 18 * 3600 else None
    timezone = {"name": tz if _TIMEZONE.match(tz) else None,
                "abbreviation": _text(data.get("timezone_abbreviation"), 16) or None,
                "utc_offset_seconds": offset}
    notes = []
    now = None
    if isinstance(current, dict):
        stamp = str(current.get("time") or "")
        valid_time = None
        if _LOCAL_TIME.match(stamp):
            try:
                valid_time = datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                valid_time = None
        interval = current.get("interval")
        now = {"time_local": valid_time,
               "interval_seconds": interval if isinstance(interval, int) and not isinstance(
                   interval, bool) and 0 < interval <= 86400 else None,
               **{v: _value(current.get(v)) for v in CURRENT_VARS}}
    else:
        notes.append("Open-Meteo did not return current conditions.")
    rows = []
    if isinstance(daily, dict):
        times = daily.get("time")
        if not isinstance(times, list):
            raise ToolError("bad_response", "Open-Meteo's daily forecast had no list of dates.")
        for i, raw_day in enumerate(times[:days]):
            try:
                day = date.fromisoformat(str(raw_day))
            except ValueError:
                notes.append("A forecast day with an invalid date was skipped.")
                continue
            row = {"date": day.isoformat(), "weekday": day.strftime("%A")}
            for var in DAILY_VARS:
                series = daily.get(var)
                row[var] = _value(series[i]) if isinstance(series, list) and i < len(
                    series) else None
            rows.append(row)
        if len(rows) < days:
            notes.append(f"Open-Meteo returned {len(rows)} of the {days} requested forecast "
                         "day(s).")
    else:
        notes.append("Open-Meteo did not return a daily forecast.")
    return {"timezone": timezone, "current": now, "daily": rows, "notes": notes,
            "units": UNIT_LABELS[units]}


def _tz_text(timezone: dict) -> str:
    parts = [timezone["name"] or "timezone not provided"]
    if timezone["abbreviation"]:
        parts.append(timezone["abbreviation"])
    offset = timezone["utc_offset_seconds"]
    if offset is not None:
        sign = "+" if offset >= 0 else "-"
        hours, minutes = divmod(abs(offset) // 60, 60)
        parts.append(f"UTC{sign}{hours:02d}:{minutes:02d}")
    return f"{parts[0]}" + (f" ({', '.join(parts[1:])})" if len(parts) > 1 else "")


def current_text(now: dict, u: dict, tz: str) -> str:
    when = now["time_local"] or NOT_PROVIDED
    interval = (f"in the last {now['interval_seconds'] // 60} minutes"
                if now["interval_seconds"] else "in the latest model interval")
    direction = (f" from {now['wind_direction_10m']:g}°"
                 if now["wind_direction_10m"] is not None else "")
    return (
        f"Current conditions (model estimate for the location, not a station observation) "
        f"valid at {when} local time, timezone {tz}. "
        f"Conditions: {_condition(now['weather_code'])}. "
        f"Temperature {_fmt(now['temperature_2m'], u['temperature'])}, feels like "
        f"{_fmt(now['apparent_temperature'], u['temperature'])}. "
        f"Relative humidity {_fmt(now['relative_humidity_2m'], '%')}. "
        f"Precipitation {interval}: {_fmt(now['precipitation'], u['precipitation'])}. "
        f"Wind {_fmt(now['wind_speed_10m'], u['wind'])}{direction}, gusts "
        f"{_fmt(now['wind_gusts_10m'], u['wind'])}."
    )


def daily_text(rows: list[dict], u: dict, tz: str) -> str:
    lines = [f"Daily forecast (predictions, not guarantees). Dates are local dates in "
             f"timezone {tz}."]
    for r in rows:
        lines.append(
            f"{r['date']} ({r['weekday']}): {_condition(r['weather_code'])}; low "
            f"{_fmt(r['temperature_2m_min'], u['temperature'])}, high "
            f"{_fmt(r['temperature_2m_max'], u['temperature'])}; precipitation total "
            f"{_fmt(r['precipitation_sum'], u['precipitation'])}; chance of precipitation up "
            f"to {_fmt(r['precipitation_probability_max'], '%')}; max wind "
            f"{_fmt(r['wind_speed_10m_max'], u['wind'])}.")
    return "\n".join(lines)


# --------------------------------------------------------------------- tool
def make_weather_tool(settings: Settings) -> ToolSpec:
    def run(args: WeatherArgs) -> dict[str, Any]:
        name = args.location.split(",")[0].strip()
        geo_params = {"name": name, "count": GEOCODE_COUNT, "language": "en", "format": "json"}
        if args.country_code:
            geo_params["countryCode"] = args.country_code
        place, others = resolve_location(_fetch_json(_geocode_get, geo_params, "geocoding"),
                                         args.location, args.country_code)
        forecast = parse_forecast(_fetch_json(_forecast_get, {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": ",".join(CURRENT_VARS), "daily": ",".join(DAILY_VARS),
            "timezone": "auto", "forecast_days": args.days, **UNIT_PARAMS[args.units],
        }, "forecast"), args.days, args.units)

        label = place_label(place)
        tz = _tz_text(forecast["timezone"])
        where = (f"Location resolved by Open-Meteo geocoding to {label} "
                 f"({place['latitude']}, {place['longitude']}).")
        if others:
            where += (" Other places with this name (not used): " + "; ".join(others) + ".")
        notes = " ".join(forecast["notes"])
        results = []
        if forecast["current"]:
            results.append({"title": f"Current weather - {label}",
                            "snippet": f"{where} " + current_text(
                                forecast["current"], forecast["units"], tz)})
        if forecast["daily"]:
            first, last = forecast["daily"][0]["date"], forecast["daily"][-1]["date"]
            results.append({"title": f"{len(forecast['daily'])}-day forecast - {label} "
                                     f"({first} to {last}, local dates)",
                            "snippet": f"{where} " + daily_text(
                                forecast["daily"], forecast["units"], tz)})
        if not results:
            raise ToolError("bad_response", "Open-Meteo's forecast response had no usable "
                            f"data. {notes}".strip())
        for i, item in enumerate(results, start=1):
            item |= {"id": i, "url": ATTRIBUTION_URL}
            if notes:
                item["snippet"] += f" Data notes: {notes}"
        return {
            "query": args.location, "source_type": "weather", "provider": "Open-Meteo",
            "attribution": ATTRIBUTION, "licence": "CC BY 4.0",
            "location": place | {"label": label}, "other_matches": others,
            "units": forecast["units"], "timezone": forecast["timezone"],
            "current": forecast["current"], "daily": forecast["daily"],
            "notes": forecast["notes"],
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results": results,
        }

    def availability(context: ToolContext | None = None) -> Availability:
        if not settings.weather_enabled:
            return Availability(AvailabilityStatus.DISABLED,
                                "disabled by OMNIFLOW_WEATHER_ENABLED")
        return Availability(AvailabilityStatus.AVAILABLE, "keyless; Open-Meteo free API")

    return ToolSpec(
        name="weather_lookup",
        description=(
            "Current weather and a daily forecast (1-16 days) for a named place, from "
            "Open-Meteo: conditions, temperature, precipitation amount and probability, and "
            "wind, with local dates and timezone. Use for weather, temperature, rain or "
            "forecast questions about a specific place. If no place is given, ask which one. "
            "Not for historical climate data."
        ),
        args_schema=WeatherArgs,
        func=run,
        category="weather",
        permission=Permission.EXTERNAL_READ,
        side_effect=SideEffect.NONE,
        timeout_seconds=2 * TIMEOUT + 5,  # two sequential requests
        max_retries=1,
        untrusted_output=True,
        availability=availability,
    )
