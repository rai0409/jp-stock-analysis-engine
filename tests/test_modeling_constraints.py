"""Tests for portfolio feasibility constraints. Deterministic, offline."""

from __future__ import annotations

from jp_stock_analysis.modeling.constraints import (
    REASON_TOO_FEW_NAMES,
    STATUS_INFEASIBLE,
    STATUS_LIQUIDITY_MISSING,
    ConstraintConfig,
    PositionBook,
    apply_constraints,
)

SECTORS = {"a": "t", "b": "t", "c": "u", "d": "u", "e": "t", "f": "u"}


def _book(adv=None):
    return PositionBook(
        long_weights={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        short_weights={"e": 0.5, "f": 0.5},
        sector_of=SECTORS,
        adv_of=adv,
    )


def _gross(weights):
    return sum(abs(w) for w in weights.values())


def test_position_cap_never_increases_absolute_exposure():
    book = _book()
    before = _gross(book.long_weights) + _gross(book.short_weights)
    result = apply_constraints(book, ConstraintConfig(max_weight_per_name=0.3))
    after = result.long.gross_after + result.short.gross_after
    assert after <= before + 1e-9
    assert max(result.long.constrained.values()) <= 0.3 + 1e-9


def test_sector_cap_when_sector_present():
    book = _book()
    result = apply_constraints(book, ConstraintConfig(max_sector_weight=0.4))
    sector_t = sum(w for n, w in result.long.constrained.items() if SECTORS[n] == "t")
    assert sector_t <= 0.4 + 1e-9
    assert "max_sector_weight" in result.applied_constraints


def test_missing_adv_returns_liquidity_missing():
    result = apply_constraints(_book(adv=None), ConstraintConfig(min_adv=10.0))
    assert result.status == STATUS_LIQUIDITY_MISSING
    assert any("not fabricating ADV" in w for w in result.warnings)


def test_min_adv_filter_removes_illiquid_names():
    adv = {"a": 100, "b": 5, "c": 100, "d": 100, "e": 100, "f": 100}
    result = apply_constraints(_book(adv=adv), ConstraintConfig(min_adv=10.0))
    assert "b" not in result.long.constrained  # illiquid name dropped
    assert "min_adv" in result.applied_constraints


def test_participation_cap_reduces_weights():
    adv = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0, "f": 1.0}
    # max_participation_rate * adv = 0.1 per name -> caps each below 0.25
    result = apply_constraints(
        _book(adv=adv), ConstraintConfig(max_participation_rate=0.1)
    )
    assert max(result.long.constrained.values()) <= 0.1 + 1e-9
    assert result.long.gross_after < result.long.gross_before


def test_infeasible_too_few_names():
    result = apply_constraints(_book(), ConstraintConfig(min_long_names=10))
    assert result.status == STATUS_INFEASIBLE
    assert result.infeasible_reason == REASON_TOO_FEW_NAMES


def test_turnover_limit_reduces_turnover():
    prior = PositionBook({"a": 1.0}, {"e": 1.0}, SECTORS)
    result = apply_constraints(
        _book(), ConstraintConfig(max_total_turnover=0.2), prior_book=prior
    )
    assert result.turnover_before > result.turnover_after
    assert result.turnover_after <= 0.2 + 1e-9
    assert "max_total_turnover" in result.applied_constraints


def test_constrained_weights_are_deterministic():
    book = _book()
    config = ConstraintConfig(max_weight_per_name=0.3, max_sector_weight=0.5)
    r1 = apply_constraints(book, config)
    r2 = apply_constraints(book, config)
    assert r1.to_dict() == r2.to_dict()


def test_max_names_keeps_largest():
    book = PositionBook({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, {}, SECTORS)
    result = apply_constraints(book, ConstraintConfig(max_long_names=2))
    assert set(result.long.constrained) == {"a", "b"}  # two largest kept
    assert "max_names" in result.long.applied


def test_research_only_disclaimer_present():
    result = apply_constraints(_book(), ConstraintConfig(max_weight_per_name=0.3))
    payload = result.to_dict()
    assert payload["research_only"] is True
    assert "not order execution" in payload["disclaimer"]
