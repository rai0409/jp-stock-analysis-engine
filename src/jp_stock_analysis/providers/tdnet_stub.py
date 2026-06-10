"""TDnet provider stub. Import-safe, no network, raises if used.

Intended future fields (TDnet timely disclosure):
- title, publish datetime, company code, document category
  (earnings revision, dividend revision, kessan tanshin), document text
"""

from __future__ import annotations

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import DisclosureDocument

_MESSAGE = (
    "TDnetProvider is a stub. TDnet retrieval is not part of this v1; place "
    "disclosure text files in a local directory instead."
)


class TDnetProvider:
    """Placeholder for a future TDnet timely-disclosure provider."""

    def get_disclosures(self, ticker: str) -> list[DisclosureDocument]:
        raise ProviderError(_MESSAGE)
