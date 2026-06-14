"""Deterministic portfolio feasibility constraints (research-only).

Applies position, liquidity/ADV, sector, and turnover constraints to a one-date
long/short weight book. This is a **research approximation of portfolio
feasibility**, NOT order execution and NOT a recommended portfolio. Liquidity/ADV
constraints require real ADV data to be meaningful — ADV is never fabricated;
when liquidity data is missing the result says so.

Ordinary infeasible cases (too few names, all names illiquid, impossible sector
cap, missing columns) return a clear ``status`` and ``warnings`` rather than
raising. Output is deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. Constraints are a research feasibility "
    "approximation, not order execution, and not a recommended portfolio."
)

STATUS_OK = "ok"
STATUS_LIQUIDITY_MISSING = "liquidity_data_missing"
STATUS_INFEASIBLE = "infeasible"
STATUS_REDUCED = "feasible_reduced"

REASON_TOO_FEW_NAMES = "too_few_names"
REASON_ALL_ILLIQUID = "all_names_fail_liquidity"
REASON_SECTOR_CAP_IMPOSSIBLE = "sector_cap_impossible"


@dataclass(frozen=True)
class ConstraintConfig:
    max_weight_per_name: float | None = None
    max_long_names: int | None = None
    max_short_names: int | None = None
    min_long_names: int | None = None
    min_short_names: int | None = None
    max_sector_weight: float | None = None
    min_adv: float | None = None
    max_participation_rate: float | None = None
    max_notional_fraction_of_adv: float | None = None
    max_total_turnover: float | None = None
    max_leg_turnover: float | None = None
    allow_unconstrained_when_liquidity_missing: bool = False

    @property
    def requires_liquidity(self) -> bool:
        return any(
            v is not None
            for v in (self.min_adv, self.max_participation_rate, self.max_notional_fraction_of_adv)
        )


@dataclass(frozen=True)
class PositionBook:
    long_weights: dict[str, float]
    short_weights: dict[str, float]
    sector_of: dict[str, str | None] = field(default_factory=dict)
    adv_of: dict[str, float] | None = None
    leg_notional: float = 1.0


@dataclass(frozen=True)
class LegResult:
    unconstrained: dict[str, float]
    constrained: dict[str, float]
    gross_before: float
    gross_after: float
    applied: list[str]
    infeasible_reason: str | None
    warnings: list[str]


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    long: LegResult
    short: LegResult
    applied_constraints: list[str]
    infeasible_reason: str | None
    turnover_before: float | None
    turnover_after: float | None
    warnings: list[str]
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "research_only": True,
            "status": self.status,
            "infeasible_reason": self.infeasible_reason,
            "applied_constraints": self.applied_constraints,
            "turnover_before": self.turnover_before,
            "turnover_after": self.turnover_after,
            "warnings": self.warnings,
            "long": {
                "unconstrained": self.long.unconstrained,
                "constrained": self.long.constrained,
                "gross_before": self.long.gross_before,
                "gross_after": self.long.gross_after,
                "applied": self.long.applied,
                "infeasible_reason": self.long.infeasible_reason,
            },
            "short": {
                "unconstrained": self.short.unconstrained,
                "constrained": self.short.constrained,
                "gross_before": self.short.gross_before,
                "gross_after": self.short.gross_after,
                "applied": self.short.applied,
                "infeasible_reason": self.short.infeasible_reason,
            },
        }


def _waterfill_cap(weights: dict[str, float], cap: Mapping[str, float], target_sum: float):
    """Cap each weight to ``cap[name]``, redistribute excess to uncapped names.

    Deterministic. If the cap budget cannot reach ``target_sum``, every name sits
    at its cap (under-invested) and a flag is returned.
    """
    if not weights:
        return {}, False
    result = dict(weights)
    for _ in range(50):
        over = {k: v for k, v in result.items() if v > cap[k] + 1e-12}
        if not over:
            break
        excess = sum(result[k] - cap[k] for k in over)
        for k in over:
            result[k] = cap[k]
        room = {k: cap[k] - result[k] for k in result if k not in over and cap[k] > result[k]}
        room_total = sum(room.values())
        if room_total <= 1e-12:
            break  # nowhere to redistribute -> under-invested
        for k, free in room.items():
            result[k] += excess * (free / room_total)
    capped_short = sum(result.values()) < target_sum - 1e-9
    return result, capped_short


def _constrain_leg(
    weights: dict[str, float],
    config: ConstraintConfig,
    book: PositionBook,
    *,
    max_names: int | None,
    min_names: int | None,
    liquidity_active: bool,
) -> LegResult:
    unconstrained = dict(weights)
    gross_before = sum(weights.values())
    applied: list[str] = []
    warnings: list[str] = []
    eligible = dict(weights)

    # 1. liquidity filter (min_adv) — requires real ADV
    if liquidity_active and book.adv_of is not None and config.min_adv is not None:
        kept = {k: v for k, v in eligible.items() if book.adv_of.get(k, 0.0) >= config.min_adv}
        if kept != eligible:
            applied.append("min_adv")
        eligible = kept
        if not eligible:
            return LegResult(
                unconstrained, {}, gross_before, 0.0, applied, REASON_ALL_ILLIQUID,
                ["all names failed the min_adv liquidity filter"],
            )

    # 2. minimum name count
    if min_names is not None and len(eligible) < min_names:
        return LegResult(
            unconstrained, {}, gross_before, 0.0, applied, REASON_TOO_FEW_NAMES,
            [f"only {len(eligible)} eligible names < min {min_names}"],
        )

    # 3. maximum name count: keep the largest weights
    if max_names is not None and len(eligible) > max_names:
        ordered = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))[:max_names]
        eligible = dict(ordered)
        applied.append("max_names")

    # renormalize the kept names back to the original leg gross
    total = sum(eligible.values())
    if total > 0:
        eligible = {k: v / total * gross_before for k, v in eligible.items()}

    # 4. per-name cap = min(position cap, liquidity caps)
    per_name_cap: dict[str, float] = {}
    for name in eligible:
        cap = config.max_weight_per_name if config.max_weight_per_name is not None else float("inf")
        if liquidity_active and book.adv_of is not None and book.leg_notional > 0:
            adv = book.adv_of.get(name, 0.0)
            if config.max_participation_rate is not None:
                cap = min(cap, config.max_participation_rate * adv / book.leg_notional)
            if config.max_notional_fraction_of_adv is not None:
                cap = min(cap, config.max_notional_fraction_of_adv * adv / book.leg_notional)
        per_name_cap[name] = cap
    if any(c < float("inf") for c in per_name_cap.values()):
        eligible, under = _waterfill_cap(eligible, per_name_cap, gross_before)
        applied.append("per_name_cap")
        if under:
            warnings.append("per-name caps prevent full investment (gross reduced)")

    # 5. sector cap
    if config.max_sector_weight is not None and book.sector_of:
        sector_total: dict[str, float] = {}
        for name, w in eligible.items():
            sector_total[book.sector_of.get(name) or "unknown"] = (
                sector_total.get(book.sector_of.get(name) or "unknown", 0.0) + w
            )
        scaled = dict(eligible)
        for sector, tot in sector_total.items():
            if tot > config.max_sector_weight + 1e-12:
                factor = config.max_sector_weight / tot
                for name in eligible:
                    if (book.sector_of.get(name) or "unknown") == sector:
                        scaled[name] *= factor
        if scaled != eligible:
            applied.append("max_sector_weight")
            warnings.append("sector cap scaled down over-weight sector(s); gross reduced")
        eligible = scaled

    gross_after = sum(eligible.values())
    return LegResult(
        unconstrained, eligible, gross_before, gross_after, applied, None, warnings
    )


def _signed(long_w: Mapping[str, float], short_w: Mapping[str, float]) -> dict[str, float]:
    book: dict[str, float] = dict(long_w)
    for k, v in short_w.items():
        book[k] = book.get(k, 0.0) - v
    return book


def _turnover(curr: Mapping[str, float], prior: Mapping[str, float]) -> float:
    names = set(curr) | set(prior)
    return 0.5 * sum(abs(curr.get(n, 0.0) - prior.get(n, 0.0)) for n in names)


def apply_constraints(
    book: PositionBook,
    config: ConstraintConfig,
    *,
    prior_book: PositionBook | None = None,
) -> ConstraintResult:
    """Apply position/liquidity/sector/turnover constraints deterministically."""
    warnings: list[str] = []
    applied: list[str] = []

    liquidity_active = config.requires_liquidity
    if config.requires_liquidity and book.adv_of is None:
        if not config.allow_unconstrained_when_liquidity_missing:
            passthrough_long = LegResult(
                dict(book.long_weights), dict(book.long_weights),
                sum(book.long_weights.values()), sum(book.long_weights.values()), [], None, [],
            )
            passthrough_short = LegResult(
                dict(book.short_weights), dict(book.short_weights),
                sum(book.short_weights.values()), sum(book.short_weights.values()), [], None, [],
            )
            return ConstraintResult(
                status=STATUS_LIQUIDITY_MISSING,
                long=passthrough_long,
                short=passthrough_short,
                applied_constraints=[],
                infeasible_reason=None,
                turnover_before=None,
                turnover_after=None,
                warnings=[
                    "liquidity/ADV constraints requested but no ADV data provided; not "
                    "fabricating ADV (set allow_unconstrained_when_liquidity_missing to "
                    "proceed without them)"
                ],
            )
        warnings.append("liquidity constraints skipped (no ADV data; explicitly allowed)")
        liquidity_active = False

    long = _constrain_leg(
        book.long_weights, config, book,
        max_names=config.max_long_names, min_names=config.min_long_names,
        liquidity_active=liquidity_active,
    )
    short = _constrain_leg(
        book.short_weights, config, book,
        max_names=config.max_short_names, min_names=config.min_short_names,
        liquidity_active=liquidity_active,
    )
    applied = sorted(set(long.applied) | set(short.applied))
    warnings.extend(long.warnings)
    warnings.extend(short.warnings)

    infeasible = long.infeasible_reason or short.infeasible_reason
    if infeasible:
        return ConstraintResult(
            status=STATUS_INFEASIBLE, long=long, short=short, applied_constraints=applied,
            infeasible_reason=infeasible, turnover_before=None, turnover_after=None,
            warnings=warnings,
        )

    # turnover (signed book vs prior)
    turnover_before = turnover_after = None
    constrained_long = dict(long.constrained)
    constrained_short = dict(short.constrained)
    if prior_book is not None:
        prior_signed = _signed(prior_book.long_weights, prior_book.short_weights)
        turnover_before = _turnover(_signed(long.unconstrained, short.unconstrained), prior_signed)
        target_signed = _signed(constrained_long, constrained_short)
        turnover_after = _turnover(target_signed, prior_signed)
        limit = config.max_total_turnover
        if limit is not None and turnover_after > limit + 1e-12 and turnover_after > 0:
            scale = limit / turnover_after
            blended = {
                n: prior_signed.get(n, 0.0)
                + scale * (target_signed.get(n, 0.0) - prior_signed.get(n, 0.0))
                for n in set(target_signed) | set(prior_signed)
            }
            constrained_long = {n: w for n, w in blended.items() if w > 1e-12}
            constrained_short = {n: -w for n, w in blended.items() if w < -1e-12}
            turnover_after = _turnover(blended, prior_signed)
            applied = sorted(set(applied) | {"max_total_turnover"})
            warnings.append("scaled book toward prior holdings to meet turnover limit")
            long = LegResult(long.unconstrained, constrained_long, long.gross_before,
                             sum(constrained_long.values()), long.applied, None, long.warnings)
            short = LegResult(short.unconstrained, constrained_short, short.gross_before,
                              sum(constrained_short.values()), short.applied, None, short.warnings)

    reduced = (
        long.gross_after < long.gross_before - 1e-9
        or short.gross_after < short.gross_before - 1e-9
    )
    status = STATUS_REDUCED if reduced else STATUS_OK
    return ConstraintResult(
        status=status, long=long, short=short, applied_constraints=applied,
        infeasible_reason=None, turnover_before=turnover_before, turnover_after=turnover_after,
        warnings=warnings,
    )


__all__ = [
    "REASON_ALL_ILLIQUID",
    "REASON_SECTOR_CAP_IMPOSSIBLE",
    "REASON_TOO_FEW_NAMES",
    "STATUS_INFEASIBLE",
    "STATUS_LIQUIDITY_MISSING",
    "STATUS_OK",
    "STATUS_REDUCED",
    "ConstraintConfig",
    "ConstraintResult",
    "LegResult",
    "PositionBook",
    "apply_constraints",
]
