"""Export a local ``ticker,date,close`` price CSV from the J-Quants provider.

This is the thinnest possible acquisition wrapper around the existing
:class:`jp_stock_analysis.providers.jquants.JQuantsProvider`. It exists so the
forward-return validation flow can obtain real prices for a fixed set of
tickers and write them in the exact raw shape ``prepare-price-csv`` expects.

Safety posture (inherited from the provider):

- **Offline-safe by default.** With ``allow_network=False`` the provider runs in
  cache-only mode: it reads ``<cache_dir>/daily_quotes/<ticker>.json`` and never
  touches the network. A live fetch happens only when ``allow_network=True`` AND
  the cache file is missing, and requires ``JQUANTS_API_KEY`` in the environment.
- **No secrets in output or errors.** The provider sends the key only in the
  ``x-api-key`` header and never includes it in error messages; this wrapper
  prints nothing but row counts and safe diagnostics.
- **No fabrication.** Prices come straight from the provider's ``PriceBar``
  rows. Tickers with no rows are reported, never invented.
- **Raw close.** The exported ``close`` is the raw close price
  (``PriceBar.close``), not adjusted close — see the caveat in
  ``docs/local_price_csv_input.md``.

This module emits no trading signals, no portfolio construction, and no
position sizing; it only acquires and reshapes price data for research.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

from jp_stock_analysis.errors import DataValidationError
from jp_stock_analysis.schemas import PriceBar


class _PriceProvider(Protocol):
    """Minimal protocol satisfied by JQuantsProvider (and test doubles)."""

    def get_prices(
        self,
        ticker: str,
        from_date: date | str | None = None,
        to_date: date | str | None = None,
    ) -> list[PriceBar]: ...


@dataclass(frozen=True)
class ExportJQuantsPricesResult:
    """Deterministic summary of a J-Quants price export run."""

    output_path: str
    tickers: list[str]
    from_date: str | None
    to_date: str | None
    total_rows_written: int
    rows_per_ticker: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def export_jquants_prices_csv(
    provider: _PriceProvider,
    tickers: Sequence[str],
    output_path: str | Path,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
) -> ExportJQuantsPricesResult:
    """Fetch daily closes per ticker and write a sorted ``ticker,date,close`` CSV.

    Raises :class:`DataValidationError` if no rows are returned for any ticker
    (a clear blocked state rather than a silent empty file). Provider errors
    (missing cache, missing credentials, API failures) propagate unchanged and
    already carry safe, secret-free messages.
    """
    requested = list(dict.fromkeys(t.strip() for t in tickers if t and t.strip()))
    if not requested:
        raise DataValidationError("no tickers requested")

    rows: list[tuple[str, str, float]] = []
    rows_per_ticker: dict[str, int] = {ticker: 0 for ticker in requested}
    for ticker in requested:
        bars = provider.get_prices(ticker, from_date=from_date, to_date=to_date)
        for bar in bars:
            rows.append((ticker, bar.date.isoformat(), bar.close))
        rows_per_ticker[ticker] = len(bars)

    empty = sorted(ticker for ticker, count in rows_per_ticker.items() if count == 0)
    if empty:
        raise DataValidationError(
            "no price rows returned for ticker(s): "
            + ", ".join(empty)
            + " (check the cache, date range, or credentials; prices are never fabricated)"
        )

    rows.sort(key=lambda item: (item[0], item[1]))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ticker", "date", "close"])
        for ticker, day, close in rows:
            writer.writerow([ticker, day, _format_close(close)])

    warnings = [
        "exported prices use raw close (PriceBar.close), not adjusted close; "
        "corporate actions are not accounted for",
    ]
    return ExportJQuantsPricesResult(
        output_path=str(out_path),
        tickers=requested,
        from_date=from_date.isoformat() if from_date else None,
        to_date=to_date.isoformat() if to_date else None,
        total_rows_written=len(rows),
        rows_per_ticker=rows_per_ticker,
        warnings=warnings,
    )


def _format_close(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


__all__ = ["ExportJQuantsPricesResult", "export_jquants_prices_csv"]
