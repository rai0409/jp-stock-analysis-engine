"""News provider stub. Import-safe, no network, raises if used.

Intended future fields:
- headline, body, published_at, source, related tickers, language
"""

from __future__ import annotations

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import DisclosureDocument

_MESSAGE = (
    "NewsProvider is a stub. News retrieval is not part of this v1 and would "
    "require an external (possibly paid) API."
)


class NewsProvider:
    """Placeholder for a future news provider."""

    def get_disclosures(self, ticker: str) -> list[DisclosureDocument]:
        raise ProviderError(_MESSAGE)
