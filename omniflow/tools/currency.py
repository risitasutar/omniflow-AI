"""`currency_convert`: exchange rates and conversions from the Frankfurter API (keyless).

One GET per lookup to https://api.frankfurter.dev/v2/rate/{BASE}/{QUOTE}[?date=YYYY-MM-DD]
through the locked-down `safe_get` (HTTPS only, fixed host allowlist, public-address check,
size cap, timeout). Response: {"date", "base", "quote", "rate"}.

Facts checked on 2026-10-14 (frankfurter.dev docs and a few read-only requests):
- No API key and no quotas (requests are rate-limited against abuse). Rates are blended
  mid-market *reference* rates from central banks and official sources, back to 1948 -
  not bank, card, remittance or live trading prices.
- Unknown currency or impossible date -> HTTP 422; a date with no data (future, or before
  coverage) -> HTTP 404. The API never converts amounts, so OmniFlow multiplies itself.

Arithmetic uses Decimal end to end (the rate is parsed from JSON as Decimal); only the
displayed converted amount is rounded. Currency names are mapped to ISO codes only when
unambiguous ("euro" -> EUR); ambiguous names ("dollar") and missing currencies become
questions for the user, never guesses.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

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

API_HOST = "api.frankfurter.dev"
RATE_URL = "https://api.frankfurter.dev/v2/rate/{base}/{quote}"
CURRENCY_HOSTS = frozenset({API_HOST})
TIMEOUT = 10.0
MAX_BYTES = 50_000
EARLIEST = date(1948, 1, 1)  # Frankfurter's documented earliest data
MAX_AMOUNT = Decimal("1000000000000")  # 1 trillion
MAX_AMOUNT_DECIMALS = 8
DISCLAIMER = ("Published reference rate (mid-market, blended from central banks and official "
              "sources). It is not a live trading quote and may differ from bank, card, "
              "remittance or exchange-bureau rates.")

_CODE = re.compile(r"^[A-Z]{3}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NAME = re.compile(r"[^a-z ]")

# Unambiguous names only. Each key is a lower-case name with letters and spaces.
ALIASES = {
    "us dollar": "USD", "us dollars": "USD", "american dollar": "USD",
    "united states dollar": "USD", "euro": "EUR", "euros": "EUR",
    "indian rupee": "INR", "indian rupees": "INR", "british pound": "GBP",
    "pound sterling": "GBP", "sterling": "GBP", "british pounds": "GBP",
    "japanese yen": "JPY", "yen": "JPY", "chinese yuan": "CNY", "yuan": "CNY",
    "renminbi": "CNY", "swiss franc": "CHF", "swiss francs": "CHF",
    "canadian dollar": "CAD", "canadian dollars": "CAD", "australian dollar": "AUD",
    "australian dollars": "AUD", "new zealand dollar": "NZD", "singapore dollar": "SGD",
    "hong kong dollar": "HKD", "pakistani rupee": "PKR", "sri lankan rupee": "LKR",
    "nepalese rupee": "NPR", "uae dirham": "AED", "emirati dirham": "AED",
    "saudi riyal": "SAR", "qatari riyal": "QAR", "south african rand": "ZAR", "rand": "ZAR",
    "mexican peso": "MXN", "brazilian real": "BRL", "real": "BRL", "russian ruble": "RUB",
    "turkish lira": "TRY", "south korean won": "KRW", "korean won": "KRW",
    "swedish krona": "SEK", "norwegian krone": "NOK", "danish krone": "DKK",
    "polish zloty": "PLN", "zloty": "PLN", "thai baht": "THB", "baht": "THB",
    "indonesian rupiah": "IDR", "rupiah": "IDR", "malaysian ringgit": "MYR",
    "ringgit": "MYR", "philippine peso": "PHP", "bangladeshi taka": "BDT", "taka": "BDT",
    "egyptian pound": "EGP", "nigerian naira": "NGN", "naira": "NGN",
    "israeli shekel": "ILS", "shekel": "ILS", "hungarian forint": "HUF", "forint": "HUF",
    "czech koruna": "CZK",
}
AMBIGUOUS = {
    "dollar": "USD, CAD, AUD, SGD, NZD, HKD", "dollars": "USD, CAD, AUD, SGD, NZD, HKD",
    "rupee": "INR, PKR, LKR, NPR", "rupees": "INR, PKR, LKR, NPR",
    "pound": "GBP, EGP", "pounds": "GBP, EGP", "peso": "MXN, PHP, ARS, CLP, COP",
    "pesos": "MXN, PHP, ARS, CLP, COP", "franc": "CHF, XOF, XAF",
    "francs": "CHF, XOF, XAF", "dirham": "AED, MAD", "riyal": "SAR, QAR, OMR",
    "krona": "SEK, ISK", "krone": "NOK, DKK", "dinar": "KWD, BHD, JOD, RSD",
    "lira": "TRY, LBP", "won": "KRW, KPW", "ruble": "RUB, BYN", "rouble": "RUB, BYN",
}


class CurrencyArgs(BaseModel):
    """Arguments for currency_convert. Values are checked inside the tool, so problems
    become clear questions for the user instead of schema errors."""

    model_config = ConfigDict(extra="forbid")
    from_currency: str | None = Field(default=None, max_length=40, description=(
        "Source currency as an ISO 4217 code such as 'USD' or 'INR' (or an unambiguous name "
        "such as 'euro'). Leave empty if the user didn't say."))
    to_currency: str | None = Field(default=None, max_length=40, description=(
        "Target currency, same format. Leave empty if the user didn't say - never assume."))
    amount: str | None = Field(default=None, max_length=40, description=(
        "Amount to convert as digits, e.g. '1000' or '49.99'. Leave empty for a rate-only "
        "question."))
    date: str | None = Field(default=None, max_length=20, description=(
        "Historical date as YYYY-MM-DD. Leave empty for the latest published rate."))

    @field_validator("amount", mode="before")
    @classmethod
    def _number_to_text(cls, value: Any) -> Any:
        """Models sometimes send numbers; keep them exact by going through their text."""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int | float):
            return repr(value)
        return value


# --------------------------------------------------------------- validation
def resolve_currency(raw: str | None, role: str) -> str:
    """ISO code for a code or unambiguous name; ToolError for missing/ambiguous/invalid."""
    if raw is None or not raw.strip():
        which = "from" if role == "source" else "to"
        raise ToolError("missing_currency", f"Which currency should I convert {which}? Please "
                        "give it as a code such as USD, EUR or INR.")
    text = " ".join(raw.split())
    if _CODE.match(text.upper()) and text.isalpha():
        return text.upper()
    name = " ".join(_NAME.sub(" ", text.lower()).split())
    if name in ALIASES:
        return ALIASES[name]
    if name in AMBIGUOUS:
        raise ToolError("ambiguous_currency", f"'{text}' could mean several currencies "
                        f"({AMBIGUOUS[name]}). Which one do you mean? Please use the 3-letter "
                        "code, e.g. USD.")
    raise ToolError("invalid_currency", f"'{text[:40]}' isn't a currency code I recognise. "
                    "Please use a 3-letter ISO code such as USD, EUR or INR.")


def parse_amount(raw: str | None) -> Decimal | None:
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip().replace(",", "").replace("_", "").replace(" ", "")
    try:
        value = Decimal(text)
    except InvalidOperation:
        value = None
    if value is None or not value.is_finite():
        raise ToolError("invalid_amount", f"'{str(raw)[:40]}' isn't a number I can convert. "
                        "Please give the amount as digits, e.g. 1000 or 49.99.")
    if value <= 0:
        raise ToolError("invalid_amount", "The amount must be greater than zero.")
    if value > MAX_AMOUNT:
        raise ToolError("invalid_amount", "The amount is too large (the limit is "
                        "1,000,000,000,000).")
    if -value.as_tuple().exponent > MAX_AMOUNT_DECIMALS:
        raise ToolError("invalid_amount", f"The amount has more than {MAX_AMOUNT_DECIMALS} "
                        "decimal places.")
    return value


def parse_date(raw: str | None, today: date) -> date | None:
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        if not _DATE.match(text):
            raise ValueError
        when = date.fromisoformat(text)
    except ValueError as exc:
        raise ToolError("invalid_date", f"'{text[:20]}' isn't a valid date. Please use "
                        "YYYY-MM-DD, e.g. 2025-01-15.") from exc
    if when > today:
        raise ToolError("invalid_date", f"{text} is in the future; exchange rates are only "
                        "published for past dates and today.")
    if when < EARLIEST:
        raise ToolError("invalid_date", "Frankfurter's exchange-rate data starts in 1948.")
    return when


# ------------------------------------------------------------------ network
def _rate_get(base: str, quote: str, params: dict, timeout: float) -> bytes:
    """The only function that calls Frankfurter (blocked in tests; mocked per test)."""
    body, _ = safe_get(RATE_URL.format(base=base, quote=quote), allowed_hosts=CURRENCY_HOSTS,
                       params=params or None, timeout=timeout, max_bytes=MAX_BYTES)
    return body


def _tool_error(exc: FetchError, base: str, quote: str, when: date | None) -> ToolError:
    if exc.category == "timeout":
        return ToolError("timeout", f"Frankfurter didn't respond within {TIMEOUT:g} seconds.",
                         retryable=True)
    if exc.category == "network":
        return ToolError("network", "Frankfurter couldn't be reached (network error).",
                         retryable=True)
    if exc.category == "rate_limited":
        return ToolError("rate_limited", "Frankfurter is rate-limiting requests (HTTP 429). "
                         "Try again later.")
    if exc.status is not None and exc.status >= 500:
        return ToolError("provider_error", f"Frankfurter is temporarily unavailable (HTTP "
                         f"{exc.status}).", retryable=True)
    if exc.status == 404:
        on = f" for {when.isoformat()}" if when else ""
        return ToolError("rate_unavailable", f"Frankfurter has no published {base}/{quote} "
                         f"rate{on}, so I can't give one. Try another date or currency pair.")
    if exc.status in (400, 422):
        return ToolError("unsupported_currency", f"Frankfurter couldn't provide a {base}/"
                         f"{quote} rate: one of these currency codes isn't supported (HTTP "
                         f"{exc.status}). Please check the codes.")
    if exc.status is not None:
        return ToolError("invalid_request", f"Frankfurter rejected the request (HTTP "
                         f"{exc.status}).")
    return ToolError("provider_error", f"Frankfurter could not be used safely ({exc.category}).")


def parse_rate(body: bytes, base: str, quote: str) -> tuple[Decimal, date]:
    """Validate {"date", "base", "quote", "rate"}; the rate stays an exact Decimal."""
    try:
        data = json.loads(body, parse_float=Decimal, parse_int=Decimal)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ToolError("bad_response", "Frankfurter returned a response that isn't valid "
                        "JSON.") from exc
    if not isinstance(data, dict):
        raise ToolError("bad_response", "Frankfurter returned a response in an unexpected "
                        "format.")
    rate, raw_date = data.get("rate"), data.get("date")
    if not isinstance(rate, Decimal) or not rate.is_finite() or rate <= 0:
        raise ToolError("bad_response", "Frankfurter's response had no valid rate.")
    try:
        when = date.fromisoformat(str(raw_date)) if _DATE.match(str(raw_date)) else None
    except ValueError:
        when = None
    if when is None:
        raise ToolError("bad_response", "Frankfurter's response had no valid date.")
    if str(data.get("base", "")).upper() != base or str(data.get("quote", "")).upper() != quote:
        raise ToolError("bad_response", "Frankfurter returned a different currency pair than "
                        "the one requested.")
    return rate, when


# ---------------------------------------------------------------- formatting
def money(value: Decimal) -> str:
    """Display value: 2 decimals, or 6 significant decimals below 1 (e.g. 0.011684)."""
    if value >= 1:
        return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"
    return f"{value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP):f}"


def plain(value: Decimal) -> str:
    """Exact value without exponent notation or trailing zeros."""
    text = f"{value:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def make_currency_tool(settings: Settings) -> ToolSpec:
    def run(args: CurrencyArgs) -> dict[str, Any]:
        base = resolve_currency(args.from_currency, "source")
        quote = resolve_currency(args.to_currency, "target")
        if base == quote:
            raise ToolError("same_currency", f"Both currencies are {base}, so there is nothing "
                            "to convert. Which other currency did you mean?")
        amount = parse_amount(args.amount)
        requested = parse_date(args.date, datetime.now(UTC).date())
        params = {"date": requested.isoformat()} if requested else {}
        try:
            body = _rate_get(base, quote, params, TIMEOUT)
        except FetchError as exc:
            raise _tool_error(exc, base, quote, requested) from exc
        rate, rate_date = parse_rate(body, base, quote)

        source_url = RATE_URL.format(base=base, quote=quote) + (
            f"?date={rate_date.isoformat()}")
        lines = [f"1 {base} = {plain(rate)} {quote} (reference rate dated "
                 f"{rate_date.isoformat()}; source: Frankfurter API)."]
        converted = None
        if amount is not None:
            converted = amount * rate  # exact; only the display is rounded
            lines.append(f"Conversion: {plain(amount)} {base} x {plain(rate)} = "
                         f"{plain(converted)} {quote} (rounded: {money(converted)} {quote}).")
        if requested and requested != rate_date:
            lines.append(f"Requested date {requested.isoformat()}; Frankfurter returned the "
                         f"rate dated {rate_date.isoformat()}.")
        elif not requested:
            lines.append("This is the latest rate Frankfurter has published.")
        lines.append(DISCLAIMER)
        title = (f"{plain(amount)} {base} to {quote}" if amount is not None
                 else f"{base} to {quote} exchange rate") + f" - reference date {rate_date}"
        return {
            "query": f"{base}/{quote}" + (f" on {requested}" if requested else ""),
            "source_type": "currency", "provider": "Frankfurter",
            "base": base, "quote": quote, "rate": plain(rate),
            "rate_date": rate_date.isoformat(),
            "requested_date": requested.isoformat() if requested else None,
            "amount": plain(amount) if amount is not None else None,
            "converted": plain(converted) if converted is not None else None,
            "converted_display": money(converted) if converted is not None else None,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results": [{"id": 1, "title": title, "url": source_url,
                         "snippet": " ".join(lines)}],
        }

    def availability(context: ToolContext | None = None) -> Availability:
        if not settings.currency_enabled:
            return Availability(AvailabilityStatus.DISABLED,
                                "disabled by OMNIFLOW_CURRENCY_ENABLED")
        return Availability(AvailabilityStatus.AVAILABLE, "keyless; Frankfurter public API")

    return ToolSpec(
        name="currency_convert",
        description=(
            "Exchange rate between two currencies, or convert an amount, using published "
            "reference rates from the Frankfurter API (central banks and official sources), "
            "latest or for a past date (YYYY-MM-DD). Returns the rate, its reference date and "
            "the converted amount. Use for any currency conversion or exchange-rate question. "
            "Not for forecasts, trading prices or investment advice."
        ),
        args_schema=CurrencyArgs,
        func=run,
        category="currency",
        permission=Permission.EXTERNAL_READ,
        side_effect=SideEffect.NONE,
        timeout_seconds=TIMEOUT + 5,
        max_retries=1,
        untrusted_output=True,
        availability=availability,
    )
