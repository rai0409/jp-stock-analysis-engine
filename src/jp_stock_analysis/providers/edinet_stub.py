"""EDINET provider stub. Import-safe, no network, raises if used.

Intended future fields (EDINET API):
- document list: docID, edinetCode, secCode, docTypeCode, periodEnd
- document content: securities report sections (business risks, MD&A) as text
"""

from __future__ import annotations

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import DisclosureDocument

_MESSAGE = (
    "EDINETProvider is a stub. EDINET document retrieval is not part of this "
    "v1; place disclosure text files in a local directory instead."
)


class EDINETProvider:
    """Placeholder for a future EDINET disclosure provider."""

    def get_disclosures(self, ticker: str) -> list[DisclosureDocument]:
        raise ProviderError(_MESSAGE)
