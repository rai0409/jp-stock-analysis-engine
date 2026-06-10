"""Provider protocols. Concrete providers are local-file based or stubs.

Future network providers (J-Quants, EDINET, TDnet, news) must implement these
protocols so the analysis pipeline stays provider-agnostic. Tests must only
ever use local providers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jp_stock_analysis.schemas import (
    CompanyMetadata,
    DisclosureDocument,
    FinancialStatement,
    PriceBar,
)


@runtime_checkable
class PriceDataProvider(Protocol):
    def get_prices(self, ticker: str) -> list[PriceBar]: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    def get_statements(self, ticker: str) -> list[FinancialStatement]: ...


@runtime_checkable
class MetadataProvider(Protocol):
    def get_metadata(self, ticker: str) -> CompanyMetadata | None: ...


@runtime_checkable
class DisclosureProvider(Protocol):
    def get_disclosures(self, ticker: str) -> list[DisclosureDocument]: ...
