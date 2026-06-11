"""Data providers: local CSV/JSON loaders, the optional cache-first J-Quants
provider, and import-safe stubs for future sources."""

from jp_stock_analysis.providers.base import (
    DisclosureProvider,
    FundamentalsProvider,
    MetadataProvider,
    PriceDataProvider,
)
from jp_stock_analysis.providers.edinet_stub import EDINETProvider
from jp_stock_analysis.providers.jquants import JQuantsProvider
from jp_stock_analysis.providers.local_csv import (
    load_company_metadata_csv,
    load_disclosure_texts,
    load_fundamentals_csv,
    load_prices_csv,
)
from jp_stock_analysis.providers.local_json import read_json, write_json
from jp_stock_analysis.providers.news_stub import NewsProvider
from jp_stock_analysis.providers.tdnet_stub import TDnetProvider

__all__ = [
    "DisclosureProvider",
    "EDINETProvider",
    "FundamentalsProvider",
    "JQuantsProvider",
    "MetadataProvider",
    "NewsProvider",
    "PriceDataProvider",
    "TDnetProvider",
    "load_company_metadata_csv",
    "load_disclosure_texts",
    "load_fundamentals_csv",
    "load_prices_csv",
    "read_json",
    "write_json",
]
