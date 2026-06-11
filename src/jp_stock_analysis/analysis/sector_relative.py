"""Sector-relative scoring: percentile ranks versus same-sector peers.

This is a purely additive analysis layer:

- ``final_score`` and all existing sub-scores are NOT modified.
- ``sector_relative_score`` is reported separately so absolute and
  sector-relative views can be compared side by side.
- Results without sector metadata, or in sectors with fewer than
  ``MIN_SECTOR_SIZE`` companies in the analyzed universe, simply get no
  sector-relative metrics (``None``) — nothing is fabricated.

Percentiles are deterministic: a value is ranked against the non-missing
values of the other same-sector companies; ties count half. 100 always means
most favorable (lower-is-better metrics — PER, PBR, risk score — are
direction-inverted before ranking).
"""

from __future__ import annotations

from jp_stock_analysis.schemas import (
    CompanyMetadata,
    SectorRelativeMetrics,
    StockAnalysisResult,
)

MIN_SECTOR_SIZE = 2
_SMALL_PEER_GROUP = 4

# percentile field -> is a higher raw value more favorable?
_HIGHER_IS_BETTER: dict[str, bool] = {
    "per_percentile": False,
    "pbr_percentile": False,
    "revenue_growth_percentile": True,
    "operating_margin_percentile": True,
    "roe_percentile": True,
    "momentum_percentile": True,
    "risk_percentile": False,
}


def _metric_values(result: StockAnalysisResult) -> dict[str, float | None]:
    valuation = result.valuation
    fundamentals = result.fundamentals
    momentum = result.momentum
    momentum_value: float | None = None
    if momentum is not None:
        momentum_value = (
            momentum.return_3m if momentum.return_3m is not None else momentum.return_6m
        )
    return {
        "per_percentile": valuation.per if valuation else None,
        "pbr_percentile": valuation.pbr if valuation else None,
        "revenue_growth_percentile": fundamentals.revenue_growth_yoy if fundamentals else None,
        "operating_margin_percentile": fundamentals.operating_margin if fundamentals else None,
        "roe_percentile": fundamentals.roe if fundamentals else None,
        "momentum_percentile": momentum_value,
        "risk_percentile": result.risks.risk_score if result.risks else None,
    }


def _percentile(value: float, peers: list[float], higher_is_better: bool) -> float:
    """Share of peers this value beats (ties count half), as 0-100."""
    beaten = 0.0
    for peer in peers:
        if value == peer:
            beaten += 0.5
        elif (value > peer) == higher_is_better:
            beaten += 1.0
    return beaten / len(peers) * 100.0


def compute_sector_relative(
    results: list[StockAnalysisResult],
    metadata: dict[str, CompanyMetadata],
) -> dict[str, SectorRelativeMetrics]:
    """Compute sector-relative metrics for every result with usable peers."""
    sectors: dict[str, list[StockAnalysisResult]] = {}
    for result in results:
        company = metadata.get(result.ticker)
        if company is not None and company.sector:
            sectors.setdefault(company.sector, []).append(result)

    computed: dict[str, SectorRelativeMetrics] = {}
    for sector, group in sectors.items():
        if len(group) < MIN_SECTOR_SIZE:
            continue
        values_by_ticker = {r.ticker: _metric_values(r) for r in group}
        for result in group:
            own = values_by_ticker[result.ticker]
            percentiles: dict[str, float | None] = {}
            missing: list[str] = []
            for field, higher_is_better in _HIGHER_IS_BETTER.items():
                value = own[field]
                peers = [
                    values_by_ticker[peer.ticker][field]
                    for peer in group
                    if peer.ticker != result.ticker
                ]
                usable_peers = [p for p in peers if p is not None]
                if value is None or not usable_peers:
                    percentiles[field] = None
                    missing.append(field)
                else:
                    percentiles[field] = round(
                        _percentile(value, usable_peers, higher_is_better), 1
                    )

            available = [v for v in percentiles.values() if v is not None]
            score = round(sum(available) / len(available), 1) if available else None
            warnings: list[str] = []
            if len(group) < _SMALL_PEER_GROUP:
                warnings.append(
                    f"small sector peer group ({len(group)} companies): ranks are coarse"
                )
            if missing:
                warnings.append("percentiles unavailable: " + ", ".join(missing))
            coverage = len(available) / len(_HIGHER_IS_BETTER)
            peer_factor = min(1.0, len(group) / _SMALL_PEER_GROUP)
            computed[result.ticker] = SectorRelativeMetrics(
                ticker=result.ticker,
                sector=sector,
                peer_count=len(group),
                sector_relative_score=score,
                warnings=warnings,
                confidence_score=round(coverage * peer_factor * 100.0, 1),
                **percentiles,
            )
    return computed


def attach_sector_relative(
    results: list[StockAnalysisResult],
    metadata: dict[str, CompanyMetadata],
) -> dict[str, SectorRelativeMetrics]:
    """Compute and attach sector-relative metrics; returns the computed map."""
    computed = compute_sector_relative(results, metadata)
    for result in results:
        result.sector_relative = computed.get(result.ticker)
    return computed
