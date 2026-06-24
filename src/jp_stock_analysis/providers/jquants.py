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

V2 endpoint facts (verified against live probes on 2026-06-13 and the official
migration guide https://jpx-jquants.com/ja/spec/migration-v1-v2):

- J-Quants V1 was retired ("J-QuantsはV2に移行しました。", HTTP 410). V2 is the
  only live version.
- the API key is sent as the ``x-api-key`` request header (a dashboard-issued
  API key under V2; the V1 ID-token / refresh-token flow is gone). It is NOT a
  Bearer token — an ``Authorization`` header is rejected by the API gateway.
- the V2 routes were restructured (verified, HTTP 200 with ``x-api-key``):

  - daily OHLC: ``/v2/equities/bars/daily``   (was ``/v1/prices/daily_quotes``)
  - financials: ``/v2/fins/summary``          (was ``/v1/fins/statements``)
  - listed master: ``/v2/equities/master``    (was ``/v1/listed/info``)

  with a ``code`` query parameter, optional ``from`` / ``to`` (``YYYY-MM-DD``),
  and ``pagination_key`` pagination. Paths/version/base are still overridable
  without code changes via environment variables (checked at construction time;
  explicit constructor arguments win over the environment):

  - ``JQUANTS_API_BASE_URL``      (default ``https://api.jquants.com``)
  - ``JQUANTS_API_VERSION``       (default ``v2``)
  - ``JQUANTS_DAILY_QUOTES_PATH`` (default ``/equities/bars/daily``)
  - ``JQUANTS_STATEMENTS_PATH``   (default ``/fins/summary``)
  - ``JQUANTS_LISTED_INFO_PATH``  (default ``/equities/master``)

- V2 response rows live under the top-level ``data`` key.
- V2 field names are abbreviated (``C`` close, ``O``/``H``/``L``, ``Vo``
  volume, ``AdjC`` adjusted close, ``Sales``/``OP``/``NP`` …). The ``_map_*``
  helpers read V2 names first and fall back to the V1 names so older synthetic
  caches keep working. Numeric fields may arrive as strings; empty = missing.

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
MIGRATION_URL = "https://jpx-jquants.com/ja/spec/migration-v1-v2"

# V2 response rows live under this top-level key; older V1 caches used a
# dataset-specific key, still accepted as a fallback (see ``_rows_from_payload``).
V2_ROWS_KEY = "data"

# dataset name -> (default V2 endpoint path, legacy V1 rows key, path override env var)
_DATASETS = {
    "daily_quotes": ("/equities/bars/daily", "daily_quotes", "JQUANTS_DAILY_QUOTES_PATH"),
    "statements": ("/fins/summary", "statements", "JQUANTS_STATEMENTS_PATH"),
    "listed_info": ("/equities/master", "info", "JQUANTS_LISTED_INFO_PATH"),
}

_PATH_ENV_VARS = ", ".join(env_name for _, _, env_name in _DATASETS.values())

# FinancialStatement field -> candidate J-Quants column names, first match wins.
# V2 (/fins/summary) abbreviated names are listed first, then the legacy V1
# (/fins/statements) names as a fallback for older caches.
# capital_expenditure has no direct J-Quants statements column and stays None.
_STATEMENT_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "revenue": ("Sales", "NetSales", "Revenue"),
    "operating_income": ("OP", "OperatingProfit"),
    "net_income": ("NP", "Profit", "NetIncome"),
    "eps": ("EPS", "EarningsPerShare"),
    "bps": ("BPS", "BookValuePerShare"),
    "dividends_per_share": ("DivAnn", "DivFY", "ResultDividendPerShareAnnual"),
    "shares_outstanding": (
        "ShOutFY",
        "AvgSh",
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        "AverageNumberOfShares",
    ),
    "total_assets": ("TA", "TotalAssets"),
    "equity": ("Eq", "Equity"),
    "operating_cash_flow": ("CFO", "CashFlowsFromOperatingActivities"),
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


def _first_value(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _normalize_jquants_code(value: Any) -> str:
    code = str(value or "").strip()
    if len(code) == 5 and code.endswith("0") and code[:4].isdigit():
        return code[:4]
    return code


def _map_daily_quote(row: dict[str, Any], ticker: str) -> PriceBar | None:
    # V2 (/equities/bars/daily) uses abbreviated names; fall back to V1 names.
    bar_date = _to_date(_first_value(row, ("Date",)))
    close = _first_float(row, ("C", "Close"))
    if bar_date is None or close is None:
        return None
    return PriceBar(
        ticker=ticker,
        date=bar_date,
        open=_first_float(row, ("O", "Open")),
        high=_first_float(row, ("H", "High")),
        low=_first_float(row, ("L", "Low")),
        close=close,
        adjusted_close=_first_float(row, ("AdjC", "AdjustmentClose")),
        volume=_first_float(row, ("Vo", "Volume")),
    )


def _map_statement(row: dict[str, Any], ticker: str) -> FinancialStatement:
    # V2 (/fins/summary): CurFYEn / CurPerEn / CurPerType / DiscDate; V1 fallback.
    period_end = _to_date(_first_value(row, ("CurFYEn", "CurrentFiscalYearEndDate"))) or _to_date(
        _first_value(row, ("CurPerEn", "CurrentPeriodEndDate"))
    )
    figures = {
        field: _first_float(row, names) for field, names in _STATEMENT_FIELD_CANDIDATES.items()
    }
    return FinancialStatement(
        ticker=ticker,
        # assumption: fiscal year labelled by the calendar year the period ends in
        fiscal_year=period_end.year if period_end else None,
        fiscal_period=_first_value(row, ("CurPerType", "TypeOfCurrentPeriod")) or None,
        source_metadata={
            "source": "jquants",
            "disclosed_date": str(_first_value(row, ("DiscDate", "DisclosedDate")) or ""),
        },
        **figures,
    )


def _map_listed_info(row: dict[str, Any], ticker: str) -> CompanyMetadata:
    # V2 (/equities/master): CoName / S33Nm / MktNm; V1 fallback names too.
    raw_code = _first_value(row, ("Code", "LocalCode", "code", "local_code"))
    company_name_en = _first_value(row, ("CoNameEn", "CompanyNameEnglish"))
    sector_17 = _first_value(row, ("S17Nm", "Sector17CodeName"))
    sector_33 = _first_value(row, ("S33Nm", "Sector33CodeName"))
    market = _first_value(row, ("MktNm", "MarketCodeName"))
    source_metadata = {"source": "jquants"}
    for key, value in (
        ("raw_code", raw_code),
        ("company_name_en", company_name_en),
        ("sector_17", sector_17),
        ("sector_33", sector_33),
        ("market", market),
    ):
        if value not in (None, ""):
            source_metadata[key] = str(value)
    return CompanyMetadata(
        ticker=ticker,
        company_name=_first_value(row, ("CoName", "CoNameEn", "CompanyName", "CompanyNameEnglish")),
        sector=_first_value(row, ("S33Nm", "S17Nm", "Sector33CodeName", "Sector17CodeName")),
        market=market,
        source_metadata=source_metadata,
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

    def date_cache_path(self, dataset: str, target_date: date | str) -> Path:
        """Deterministic cache file location for one dataset/date query."""
        day = _to_date(target_date)
        if day is None:
            raise ProviderError(f"invalid J-Quants date query: {target_date!r}")
        return self.cache_dir / f"{dataset}_by_date" / f"{day.isoformat()}.json"

    def all_listed_info_cache_path(self) -> Path:
        """Deterministic cache file location for the full listed master snapshot."""
        return self.cache_dir / "listed_info" / "_all.json"

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

    def fetch_daily_bars_by_date(
        self,
        target_date: date | str,
        *,
        allow_network: bool = False,
    ) -> list[PriceBar]:
        """Fetch/cache all daily bars for one trading date.

        Offline by default: cached ``daily_quotes_by_date/YYYY-MM-DD.json`` rows
        are used when present. A live date query is made only when
        ``allow_network`` is true. Rows are mapped with J-Quants ``Code``
        normalized from the V2 5-digit form (e.g. ``72030`` -> ``7203``).
        """
        day = _to_date(target_date)
        if day is None:
            raise ProviderError(f"invalid J-Quants date query: {target_date!r}")
        rows = self._load_rows_by_date(
            "daily_quotes", day, allow_network=allow_network
        )
        bars = [
            bar
            for row in rows
            if (bar := _map_daily_quote(row, _normalize_jquants_code(row.get("Code"))))
            is not None
        ]
        bars.sort(key=lambda bar: (bar.ticker, bar.date))
        return bars

    def get_statements(self, ticker: str) -> list[FinancialStatement]:
        rows = self._load_rows("statements", ticker, {})
        statements = [_map_statement(row, ticker) for row in rows]
        statements.sort(key=lambda s: (s.fiscal_year is None, s.fiscal_year or 0))
        return statements

    def get_metadata(self, ticker: str) -> CompanyMetadata | None:
        rows = self._load_rows("listed_info", ticker, {})
        return _map_listed_info(rows[0], ticker) if rows else None

    def get_all_metadata(self, *, allow_network: bool = False) -> dict[str, CompanyMetadata]:
        """Fetch/cache the full listed master snapshot and map it by normalized code.

        This is still provider-backed: live access goes through the same
        ``_fetch_rows_by_query`` path and ``x-api-key`` auth as per-code
        ``get_metadata`` calls. Offline callers may use a cached
        ``listed_info/_all.json`` snapshot.
        """
        path = self.all_listed_info_cache_path()
        if path.exists():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"invalid J-Quants cache file {path}: {exc}") from exc
            if not isinstance(rows, list):
                raise ProviderError(f"invalid J-Quants cache file {path}: expected a JSON list")
        else:
            if not allow_network:
                raise ProviderError(
                    f"no cached J-Quants listed_info master snapshot (expected {path}); "
                    "full-master mode never fetches unless --allow-network is set."
                )
            rows = self._fetch_rows_by_query("listed_info", {})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        mapped: dict[str, CompanyMetadata] = {}
        for row in rows:
            raw_code = _first_value(row, ("Code", "LocalCode", "code", "local_code"))
            code = _normalize_jquants_code(raw_code)
            if code:
                mapped[code] = _map_listed_info(row, code)
        return mapped

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

    def _load_rows_by_date(
        self,
        dataset: str,
        target_date: date,
        *,
        allow_network: bool,
    ) -> list[dict[str, Any]]:
        path = self.date_cache_path(dataset, target_date)
        if path.exists():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"invalid J-Quants cache file {path}: {exc}") from exc
            if not isinstance(rows, list):
                raise ProviderError(f"invalid J-Quants cache file {path}: expected a JSON list")
            return rows
        if not allow_network:
            raise ProviderError(
                f"no cached J-Quants {dataset} data for {target_date.isoformat()} "
                f"(expected {path}); date mode never fetches unless --allow-network is set."
            )
        rows = self._fetch_rows_by_query(dataset, {"date": target_date.isoformat()})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rows

    def _fetch_rows(self, dataset: str, code: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        query = {"code": code, **{k: v for k, v in params.items() if v}}
        return self._fetch_rows_by_query(dataset, query)

    def _fetch_rows_by_query(
        self, dataset: str, query_params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self._api_key:
            raise ProviderError(
                f"live J-Quants fetch requested but the {ENV_API_KEY} environment "
                "variable is not set; export it or use cached data."
            )
        legacy_rows_key = _DATASETS[dataset][1]
        rows: list[dict[str, Any]] = []
        pagination_key: str | None = None
        while True:
            query = {k: v for k, v in query_params.items() if v}
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
            rows.extend(_rows_from_payload(payload, legacy_rows_key))
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                return rows


def _rows_from_payload(payload: dict[str, Any], legacy_key: str) -> list[dict[str, Any]]:
    """Extract data rows from a J-Quants response.

    V2 nests rows under ``data``; older V1 responses used a dataset-specific key
    (``daily_quotes`` / ``statements`` / ``info``), still accepted as a fallback.
    """
    rows = payload.get(V2_ROWS_KEY)
    if rows is None:
        rows = payload.get(legacy_key, [])
    return rows or []


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
    # V1 retired / migrated to V2 (HTTP 410 "J-QuantsはV2に移行しました。").
    if exc.code == 410 or "移行" in body or "migration" in lowered:
        return (
            f"J-Quants V1 has been retired (HTTP {exc.code}) at {url}: the service migrated "
            f"to V2. Use the V2 defaults (version {DEFAULT_API_VERSION}, daily quotes path "
            "/equities/bars/daily) or set the correct override env vars. Migration guide: "
            f"{MIGRATION_URL}. Server response: {body}"
        )
    if "endpoint does not exist" in lowered:
        return (
            f"J-Quants endpoint not found (HTTP {exc.code}) at {url}: the configured API "
            "version or path does not match the service. The verified V2 daily-quotes path "
            f"is /equities/bars/daily. Check the official spec ({SPEC_URL}) and override via "
            f"{ENV_BASE_URL}, {ENV_API_VERSION}, or {_PATH_ENV_VARS}. Server response: {body}"
        )
    # Plan / subscription window: data requested outside the subscribed date range.
    if "subscription covers" in lowered or "check other plans" in lowered:
        return (
            f"J-Quants plan/date-coverage limit (HTTP {exc.code}) at {url}: the requested "
            "date range is outside your subscription's covered dates. Narrow the date range "
            f"to the covered window or upgrade the plan. Server response: {body}"
        )
    if "authorization" in lowered and ("malformed" in lowered or "bearer" in lowered):
        return (
            f"J-Quants rejected the Authorization header (HTTP {exc.code}) at {url}: the "
            "API key is not a Bearer token. This provider sends the key as the x-api-key "
            "header; do not place it in an Authorization header. "
            f"Server response: {body}"
        )
    if exc.code in (401, 403):
        return (
            f"J-Quants authentication/permission failure (HTTP {exc.code}) at {url}: the "
            f"{ENV_API_KEY} may be missing, invalid, or lack access to this dataset. The key "
            "is sent only as the x-api-key header and is never logged. "
            f"Server response: {body or '(empty)'}"
        )
    return (
        f"J-Quants request failed (HTTP {exc.code}) at {url}. "
        f"Server response: {body or '(empty)'}"
    )


def _urllib_get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)
