"""Optional J-Quants V2 API provider with local cache-first loading.

Design:

- **Cache-first.** ``get_*`` methods read deterministic JSON cache files under
  ``cache_dir`` (default ``.cache/jquants/``) and work fully offline.
- **Live fetch is explicit opt-in.** Only when the provider was constructed
  with ``live=True`` AND the cache file is missing does it call the API; the
  response rows are then written to the cache for future offline runs.
- **Import-safe without credentials.** ``JQUANTS_API_KEY`` is only required at
  the moment a live fetch actually happens; constructing the provider and
  reading caches never needs it.
- Rows map into the existing ``PriceBar`` / ``FinancialStatement`` /
  ``CompanyMetadata`` schemas. Missing fields stay ``None`` — never fabricated.

Adapter assumptions (isolated in the ``_map_*`` helpers and ``_DATASETS``
table; verify against the official spec at https://jpx-jquants.com/spec/
before live use):

- the API key is sent as the ``x-api-key`` request header. It is NOT a
  Bearer token: a live probe showed ``Authorization: Bearer <api-key>`` is
  rejected as malformed.
- default endpoints ``/v2/prices/daily_quotes``, ``/v2/fins/statements``,
  ``/v2/listed/info`` with a ``code`` query parameter and ``pagination_key``
  pagination. A live probe of ``/v2/prices/daily_quotes`` returned HTTP 403
  "The requested endpoint does not exist", so the exact version/paths are
  UNVERIFIED — they are configurable without code changes via environment
  variables (checked at construction time; explicit constructor arguments
  win over the environment):

  - ``JQUANTS_API_BASE_URL``      (default ``https://api.jquants.com``)
  - ``JQUANTS_API_VERSION``       (default ``v2``)
  - ``JQUANTS_DAILY_QUOTES_PATH`` (default ``/prices/daily_quotes``)
  - ``JQUANTS_STATEMENTS_PATH``   (default ``/fins/statements``)
  - ``JQUANTS_LISTED_INFO_PATH``  (default ``/listed/info``)

- response rows live under ``daily_quotes`` / ``statements`` / ``info``
- numeric fields may arrive as strings; empty strings mean missing

Error messages never contain the API key. J-Quants raw data must not be
redistributed; cache files are for local use only and are not committed
(tests use synthetic cache fixtures).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import CompanyMetadata, FinancialStatement, PriceBar

ENV_API_KEY = "JQUANTS_API_KEY"
ENV_BASE_URL = "JQUANTS_API_BASE_URL"
ENV_API_VERSION = "JQUANTS_API_VERSION"
DEFAULT_BASE_URL = "https://api.jquants.com"
DEFAULT_API_VERSION = "v2"
DEFAULT_CACHE_DIR = ".cache/jquants"
SPEC_URL = "https://jpx-jquants.com/spec/"

# dataset name -> (default endpoint path, response rows key, path override env var)
_DATASETS = {
    "daily_quotes": ("/prices/daily_quotes", "daily_quotes", "JQUANTS_DAILY_QUOTES_PATH"),
    "statements": ("/fins/statements", "statements", "JQUANTS_STATEMENTS_PATH"),
    "listed_info": ("/listed/info", "info", "JQUANTS_LISTED_INFO_PATH"),
}

_PATH_ENV_VARS = ", ".join(env_name for _, _, env_name in _DATASETS.values())

# FinancialStatement field -> candidate J-Quants column names, first match wins.
# capital_expenditure has no direct J-Quants statements column and stays None.
_STATEMENT_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "revenue": ("NetSales", "Revenue"),
    "operating_income": ("OperatingProfit",),
    "net_income": ("Profit", "NetIncome"),
    "eps": ("EarningsPerShare",),
    "bps": ("BookValuePerShare",),
    "dividends_per_share": ("ResultDividendPerShareAnnual",),
    "shares_outstanding": (
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        "AverageNumberOfShares",
    ),
    "total_assets": ("TotalAssets",),
    "equity": ("Equity",),
    "operating_cash_flow": ("CashFlowsFromOperatingActivities",),
}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def _first_float(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _to_float(row.get(name))
        if value is not None:
            return value
    return None


def _map_daily_quote(row: dict[str, Any], ticker: str) -> PriceBar | None:
    bar_date = _to_date(row.get("Date"))
    close = _to_float(row.get("Close"))
    if bar_date is None or close is None:
        return None
    return PriceBar(
        ticker=ticker,
        date=bar_date,
        open=_to_float(row.get("Open")),
        high=_to_float(row.get("High")),
        low=_to_float(row.get("Low")),
        close=close,
        adjusted_close=_to_float(row.get("AdjustmentClose")),
        volume=_to_float(row.get("Volume")),
    )


def _map_statement(row: dict[str, Any], ticker: str) -> FinancialStatement:
    period_end = _to_date(row.get("CurrentFiscalYearEndDate")) or _to_date(
        row.get("CurrentPeriodEndDate")
    )
    figures = {
        field: _first_float(row, names) for field, names in _STATEMENT_FIELD_CANDIDATES.items()
    }
    return FinancialStatement(
        ticker=ticker,
        # assumption: fiscal year labelled by the calendar year the period ends in
        fiscal_year=period_end.year if period_end else None,
        fiscal_period=row.get("TypeOfCurrentPeriod") or None,
        source_metadata={
            "source": "jquants",
            "disclosed_date": str(row.get("DisclosedDate") or ""),
        },
        **figures,
    )


def _map_listed_info(row: dict[str, Any], ticker: str) -> CompanyMetadata:
    return CompanyMetadata(
        ticker=ticker,
        company_name=row.get("CompanyName") or row.get("CompanyNameEnglish") or None,
        sector=row.get("Sector33CodeName") or row.get("Sector17CodeName") or None,
        market=row.get("MarketCodeName") or None,
        source_metadata={"source": "jquants"},
    )


class JQuantsProvider:
    """Cache-first J-Quants provider implementing the local provider protocols.

    ``ticker`` arguments are passed through as the J-Quants ``code`` query
    parameter and used verbatim for cache file names and output schemas, so
    4-digit (``7203``) and 5-digit (``72030``) codes both work consistently.
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        live: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        endpoint_paths: dict[str, str] | None = None,
        http_get: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.live = live
        self._api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        self.base_url = (
            base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_version = (
            api_version or os.environ.get(ENV_API_VERSION) or DEFAULT_API_VERSION
        ).strip("/")
        overrides = endpoint_paths or {}
        self._endpoint_paths: dict[str, str] = {}
        for dataset, (default_path, _rows_key, env_name) in _DATASETS.items():
            path = overrides.get(dataset) or os.environ.get(env_name) or default_path
            self._endpoint_paths[dataset] = path if path.startswith("/") else f"/{path}"
        self._http_get = http_get or _urllib_get_json

    def endpoint_url(self, dataset: str) -> str:
        """Resolved endpoint URL (base / version / path, all overridable)."""
        return f"{self.base_url}/{self.api_version}{self._endpoint_paths[dataset]}"

    def cache_path(self, dataset: str, code: str) -> Path:
        """Deterministic cache file location for one dataset/code pair."""
        return self.cache_dir / dataset / f"{code}.json"

    def get_prices(
        self,
        ticker: str,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
    ) -> list[PriceBar]:
        start = _to_date(from_date)
        end = _to_date(to_date)
        params = {
            "from": start.isoformat() if start else None,
            "to": end.isoformat() if end else None,
        }
        rows = self._load_rows("daily_quotes", ticker, params)
        bars = [bar for bar in (_map_daily_quote(row, ticker) for row in rows) if bar is not None]
        if start is not None:
            bars = [bar for bar in bars if bar.date >= start]
        if end is not None:
            bars = [bar for bar in bars if bar.date <= end]
        bars.sort(key=lambda bar: bar.date)
        return bars

    def get_statements(self, ticker: str) -> list[FinancialStatement]:
        rows = self._load_rows("statements", ticker, {})
        statements = [_map_statement(row, ticker) for row in rows]
        statements.sort(key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0))
        return statements

    def get_metadata(self, ticker: str) -> CompanyMetadata | None:
        rows = self._load_rows("listed_info", ticker, {})
        return _map_listed_info(rows[0], ticker) if rows else None

    def _load_rows(self, dataset: str, code: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        path = self.cache_path(dataset, code)
        if path.exists():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"invalid J-Quants cache file {path}: {exc}") from exc
            if not isinstance(rows, list):
                raise ProviderError(f"invalid J-Quants cache file {path}: expected a JSON list")
            return rows
        if not self.live:
            raise ProviderError(
                f"no cached J-Quants {dataset} data for {code} (expected {path}); "
                "cache mode never fetches. Run with the live provider explicitly "
                "enabled to fetch and populate the cache."
            )
        rows = self._fetch_rows(dataset, code, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rows

    def _fetch_rows(self, dataset: str, code: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._api_key:
            raise ProviderError(
                f"live J-Quants fetch requested but the {ENV_API_KEY} environment "
                "variable is not set; export it or use cached data."
            )
        rows_key = _DATASETS[dataset][1]
        rows: list[dict[str, Any]] = []
        pagination_key: str | None = None
        while True:
            query = {"code": code, **{k: v for k, v in params.items() if v}}
            if pagination_key:
                query["pagination_key"] = pagination_key
            url = f"{self.endpoint_url(dataset)}?{urllib.parse.urlencode(query)}"
            try:
                # the key travels only in the x-api-key header (it is not a Bearer token)
                payload = self._http_get(url, {"x-api-key": self._api_key})
            except urllib.error.HTTPError as exc:
                raise ProviderError(_describe_http_error(url, exc)) from exc
            except OSError as exc:
                raise ProviderError(f"J-Quants request failed for {url}: {exc}") from exc
            rows.extend(payload.get(rows_key, []))
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                return rows


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except (OSError, ValueError):
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")[:300]


def _describe_http_error(url: str, exc: urllib.error.HTTPError) -> str:
    """Classify HTTP failures into actionable messages. Never includes secrets."""
    body = _read_error_body(exc)
    lowered = body.lower()
    if "endpoint does not exist" in lowered:
        return (
            f"J-Quants endpoint not found (HTTP {exc.code}) at {url}: the configured API "
            "version or path does not match the service (observed for the default "
            f"/{DEFAULT_API_VERSION} paths in a live probe). Check the official spec "
            f"({SPEC_URL}) and override via {ENV_BASE_URL}, {ENV_API_VERSION}, or "
            f"{_PATH_ENV_VARS}. Server response: {body}"
        )
    if "authorization" in lowered and ("malformed" in lowered or "bearer" in lowered):
        return (
            f"J-Quants rejected the Authorization header (HTTP {exc.code}) at {url}: the "
            "API key is not a Bearer token. This provider sends the key as the x-api-key "
            "header; do not place it in an Authorization header. "
            f"Server response: {body}"
        )
    return (
        f"J-Quants request failed (HTTP {exc.code}) at {url}. "
        f"Server response: {body or '(empty)'}"
    )


def _urllib_get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)
