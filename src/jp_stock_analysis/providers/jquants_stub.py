"""J-Quants provider stub. Import-safe, no network, raises if used.

Intended future fields (J-Quants API):
- daily quotes: Date, Code, Open, High, Low, Close, AdjustmentClose, Volume
- statements: DisclosedDate, TypeOfDocument, NetSales, OperatingProfit,
  Profit, EarningsPerShare, BookValuePerShare, Equity, TotalAssets
"""

from __future__ import annotations

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import FinancialStatement, PriceBar

_MESSAGE = (
    "JQuantsProvider is a stub. Network access and a J-Quants API token are "
    "not part of this v1; use local CSV providers instead."
)


class JQuantsProvider:
    """Placeholder for a future J-Quants price/fundamentals provider."""

    def get_prices(self, ticker: str) -> list[PriceBar]:
        raise ProviderError(_MESSAGE)

    def get_statements(self, ticker: str) -> list[FinancialStatement]:
        raise ProviderError(_MESSAGE)
