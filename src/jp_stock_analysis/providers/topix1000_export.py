"""File-based provider for topix1000_disclosure_platform EDINET exports.

topix1000_disclosure_platform owns EDINET ingestion (PostgreSQL, raw
archives); this engine only reads the deterministic JSON files it exports.
No network access, no database access, no PostgreSQL dependency.

Expected export directory layout::

    export_dir/
        index.json                  # {"documents": [{"doc_id": ..., "path": ...}, ...]}
        disclosures/<doc_id>.json   # one JSON per document

``index.json`` may also be a bare list of entries; an entry may be a string
path. An entry without ``path`` defaults to ``disclosures/<doc_id>.json``.

Per-document fields used (unknown fields are ignored): ``doc_id``,
``target_date``, ``edinet_code``, ``ticker``/``sec_code``, ``company_name``,
``filing_type_key``, ``document_type``, ``form_code``, ``doc_type_code``,
``fiscal_year``, ``source_paths``, ``xbrl_facts_summary``, ``text``.

``text`` may be ``null`` (the platform exports metadata before text
extraction exists). Such documents become metadata-only
``DisclosureDocument``s with an explicit warning so downstream disclosure
analysis degrades to low confidence — text is never fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jp_stock_analysis.errors import ProviderError
from jp_stock_analysis.schemas import DisclosureDocument

INDEX_FILENAME = "index.json"
_DEFAULT_DOC_SUBDIR = "disclosures"

# scalar per-document fields copied verbatim into source_metadata
_METADATA_FIELDS = (
    "doc_id",
    "target_date",
    "edinet_code",
    "sec_code",
    "filing_type_key",
    "form_code",
    "doc_type_code",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ProviderError(f"topix1000 export file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid JSON in topix1000 export file {path}: {exc}") from exc


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ticker(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (ticker, warning). Prefers ``ticker``, then ``sec_code``.

    EDINET ``secCode`` is the 4-digit listing code padded with a trailing
    zero ("72030" -> "7203"); plain 4-digit codes pass through unchanged.
    """
    ticker = payload.get("ticker")
    if ticker:
        return str(ticker), None
    sec_code = payload.get("sec_code")
    if sec_code:
        code = str(sec_code).strip()
        if len(code) == 5 and code.isdigit() and code.endswith("0"):
            return code[:4], None
        return code, None
    edinet_code = payload.get("edinet_code")
    if edinet_code:
        return (
            str(edinet_code),
            "no ticker/sec_code in export document; keyed by EDINET code",
        )
    return None, None


class Topix1000ExportProvider:
    """Reads a topix1000_disclosure_platform export directory.

    Construction loads and validates ``index.json`` eagerly so configuration
    errors surface immediately; per-document JSON is read by
    ``load_documents()`` / ``get_disclosures()``. Non-fatal issues (empty
    export, duplicate tickers, missing text) accumulate in ``warnings``.
    """

    def __init__(self, export_dir: str | Path) -> None:
        self.export_dir = Path(export_dir)
        self.warnings: list[str] = []
        if not self.export_dir.is_dir():
            raise ProviderError(f"topix1000 export directory not found: {self.export_dir}")
        self._doc_paths = self._resolve_index(_read_json(self.export_dir / INDEX_FILENAME))
        if not self._doc_paths:
            self.warnings.append(
                f"topix1000 export index lists no documents: {self.export_dir / INDEX_FILENAME}"
            )

    def _resolve_index(self, index: Any) -> list[Path]:
        entries = index.get("documents") if isinstance(index, dict) else index
        if not isinstance(entries, list):
            raise ProviderError(
                "topix1000 export index.json must be a list of documents or an object "
                f"with a 'documents' list: {self.export_dir / INDEX_FILENAME}"
            )
        paths: list[Path] = []
        for position, entry in enumerate(entries):
            if isinstance(entry, str):
                relative = entry
            elif isinstance(entry, dict) and entry.get("path"):
                relative = str(entry["path"])
            elif isinstance(entry, dict) and entry.get("doc_id"):
                relative = f"{_DEFAULT_DOC_SUBDIR}/{entry['doc_id']}.json"
            else:
                raise ProviderError(
                    f"topix1000 export index entry {position} needs 'path' or 'doc_id'"
                )
            paths.append(self.export_dir / relative)
        return paths

    def _map_document(self, path: Path) -> DisclosureDocument:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ProviderError(f"topix1000 export document must be a JSON object: {path}")

        warnings: list[str] = []
        ticker, ticker_warning = _normalize_ticker(payload)
        if ticker is None:
            raise ProviderError(
                f"topix1000 export document has no ticker, sec_code, or edinet_code: {path}"
            )
        if ticker_warning:
            warnings.append(ticker_warning)

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            text = ""
            warnings.append(
                "export document has no extracted text; metadata-only document "
                "(text is never fabricated)"
            )

        source_metadata = {
            field: str(payload[field])
            for field in _METADATA_FIELDS
            if payload.get(field) is not None
        }
        source_paths = payload.get("source_paths")
        if isinstance(source_paths, dict):
            for key, value in sorted(source_paths.items()):
                if value is not None:
                    source_metadata[f"source_path.{key}"] = str(value)
        facts_summary = payload.get("xbrl_facts_summary")
        if isinstance(facts_summary, dict):
            for key, value in sorted(facts_summary.items()):
                if value is not None:
                    source_metadata[f"xbrl.{key}"] = str(value)

        return DisclosureDocument(
            ticker=ticker,
            text=text,
            document_type=payload.get("document_type") or payload.get("filing_type_key"),
            fiscal_year=_to_int(payload.get("fiscal_year")),
            source=str(path),
            doc_id=payload.get("doc_id"),
            edinet_code=payload.get("edinet_code"),
            company_name=payload.get("company_name"),
            warnings=warnings,
            source_metadata=source_metadata,
        )

    def load_all(self) -> list[DisclosureDocument]:
        """Load every indexed document in index order."""
        return [self._map_document(path) for path in self._doc_paths]

    def load_documents(self) -> dict[str, DisclosureDocument]:
        """Load documents keyed by ticker for the analysis pipeline.

        When one ticker has several documents, the latest wins
        (``target_date`` then ``doc_id``, both lexicographic) and a warning
        records the doc ids that were set aside.
        """
        grouped: dict[str, list[DisclosureDocument]] = {}
        for document in self.load_all():
            grouped.setdefault(document.ticker, []).append(document)
        selected: dict[str, DisclosureDocument] = {}
        for ticker, documents in grouped.items():
            documents.sort(
                key=lambda d: (
                    d.source_metadata.get("target_date", ""),
                    d.doc_id or "",
                )
            )
            selected[ticker] = documents[-1]
            if len(documents) > 1:
                skipped = ", ".join(d.doc_id or d.source or "?" for d in documents[:-1])
                self.warnings.append(
                    f"ticker {ticker}: multiple export documents; "
                    f"using {documents[-1].doc_id or documents[-1].source}, skipping {skipped}"
                )
        return selected

    def get_disclosures(self, ticker: str) -> list[DisclosureDocument]:
        """DisclosureProvider protocol: all documents for one ticker."""
        return [document for document in self.load_all() if document.ticker == ticker]
