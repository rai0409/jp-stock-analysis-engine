"""Extended, trading-oriented feature set for the GBDT ranking models.

Adds ~40 widely available features on top of the 18 core factors
(``factors.ALL_FACTORS``). Every feature is computed **as of the decision date**
from information public at that time (bars dated <= decision date, statements
already disclosed, margin balances already published, *scheduled* earnings
dates). Missing inputs yield ``None`` — never a fabricated number.

Families (see :data:`TRADING_FEATURE_GROUPS`):

- ``price``      adjusted-close returns, 12-1 momentum, moving-average deviation
                 (5/25/75), distance to the 52-week high, RSI(14), MACD
                 histogram, Bollinger %b, 20-day realised volatility
- ``candle``     candlestick shape from split-adjusted OHLC: body / upper- and
                 lower-shadow ratios, overnight gap, intraday return, 5-day mean
                 body ratio, ATR(14), Parkinson and Garman-Klass volatility
- ``liquidity``  20-day traded value (yen, log), 5/20-day traded-value ratio,
                 Amihud illiquidity, share turnover
- ``valuation``  forecast earnings yield (会社予想 EPS / price), forecast
                 dividend yield, FCF yield, log market cap
- ``quality``    gross profitability (Novy-Marx), accruals
- ``events``     guidance revision, earnings surprise vs prior guidance, days
                 since the last disclosure, post-disclosure drift, days to the
                 next *scheduled* earnings date
- ``supply``     margin long/short balance ratio and long balance vs volume
- ``risk``       60-day beta and idiosyncratic volatility vs an equal-weighted
                 universe market proxy

Price conventions: returns / candles / volatility use **split-adjusted** OHLC
(``open/high/low`` scaled by ``adjusted_close / close``); market cap, yields and
traded value use **unadjusted** close (paired with as-reported shares and raw
volume). Traded value (close x volume) is split-invariant, so liquidity ratios
use it instead of share volume.

The models rank every feature within each decision date, so market-wide
constants (e.g. the index level) carry no cross-sectional information and are
deliberately not included here; regime information belongs to the timing /
research layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from jp_stock_analysis.schemas import FinancialStatement, MarginBalance, PriceBar

TRADING_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "ret_1d", "ret_5d", "mom_12_1", "ma_dev_5", "ma_dev_25", "ma_dev_75",
        "high_52w_ratio", "rsi_14", "macd_hist_pct", "bb_pctb_20", "realized_vol_20",
    ),
    "candle": (
        "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "gap_1d",
        "intraday_ret_1d", "avg_body_ratio_5", "atr_14_pct", "parkinson_vol_20",
        "garman_klass_vol_20",
    ),
    "liquidity": (
        "log_traded_value_20d", "traded_value_ratio_5_20", "amihud_illiq_20",
        "share_turnover_20",
    ),
    "valuation": (
        "forecast_earnings_yield", "forecast_dividend_yield", "fcf_yield", "log_market_cap",
    ),
    "quality": ("gross_profitability", "accruals"),
    "events": (
        "forecast_eps_revision", "earnings_surprise", "days_since_disclosure",
        "post_disclosure_return", "days_to_next_earnings",
    ),
    "supply": ("margin_ratio_log", "margin_long_to_volume"),
    "risk": ("beta_60d", "idio_vol_60d"),
}

TRADING_FEATURES: tuple[str, ...] = tuple(
    f for group in TRADING_FEATURE_GROUPS.values() for f in group
)

TRADING_DAYS_PER_YEAR = 252
# A scheduled earnings date is only treated as known when it is at most this
# many calendar days ahead (JPX/companies publish schedules ~1 month before).
MAX_KNOWN_EARNINGS_LEAD_DAYS = 45
# Margin balances are published a few business days after the reference date;
# when the publication date is unknown, assume this conservative lag.
DEFAULT_MARGIN_PUBLICATION_LAG_DAYS = 7


@dataclass(frozen=True)
class _Ohlc:
    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float  # adjusted
    raw_close: float
    volume: float | None


def adjusted_ohlc(bars: Sequence[PriceBar]) -> list[_Ohlc]:
    """Split-adjusted OHLC: O/H/L scaled by the bar's ``adjusted_close / close``."""
    out: list[_Ohlc] = []
    for bar in sorted(bars, key=lambda b: b.date):
        raw = float(bar.close)
        factor = (
            float(bar.adjusted_close) / raw
            if (bar.adjusted_close is not None and raw != 0)
            else 1.0
        )
        def adj(value: float | None, f: float = factor) -> float | None:
            return None if value is None else float(value) * f
        out.append(
            _Ohlc(
                date=bar.date,
                open=adj(bar.open),
                high=adj(bar.high),
                low=adj(bar.low),
                close=raw * factor,
                raw_close=raw,
                volume=None if bar.volume is None else float(bar.volume),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# small numeric helpers (pure python, deterministic)
# --------------------------------------------------------------------------- #
def _pct(x: float | None) -> float | None:
    return None if x is None else x * 100.0


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _ema(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    for v in values:
        out.append(v if not out else alpha * v + (1 - alpha) * out[-1])
    return out


def _returns(closes: Sequence[float]) -> list[float]:
    return [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def _price_features(rows: Sequence[_Ohlc]) -> dict[str, float | None]:
    c = [r.close for r in rows]
    n = len(c)
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["price"])
    if n >= 2:
        f["ret_1d"] = _pct(_ratio(c[-1], c[-2]) - 1.0) if c[-2] else None
    if n >= 6 and c[-6]:
        f["ret_5d"] = _pct(c[-1] / c[-6] - 1.0)
    if n >= 253 and c[-253]:
        f["mom_12_1"] = _pct(c[-22] / c[-253] - 1.0)
    for window, name in ((5, "ma_dev_5"), (25, "ma_dev_25"), (75, "ma_dev_75")):
        if n >= window:
            ma = sum(c[-window:]) / window
            f[name] = _pct(c[-1] / ma - 1.0) if ma else None
    if n >= 20:
        highs = [r.high if r.high is not None else r.close for r in rows[-252:]]
        top = max(highs)
        f["high_52w_ratio"] = c[-1] / top if top else None
    if n >= 15:
        diffs = [c[i] - c[i - 1] for i in range(n - 14, n)]
        gain = sum(d for d in diffs if d > 0) / 14
        loss = sum(-d for d in diffs if d < 0) / 14
        f["rsi_14"] = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
    if n >= 35:
        macd = [a - b for a, b in zip(_ema(c, 12), _ema(c, 26), strict=True)]
        signal = _ema(macd, 9)
        f["macd_hist_pct"] = _pct((macd[-1] - signal[-1]) / c[-1]) if c[-1] else None
    if n >= 20:
        window = c[-20:]
        ma = sum(window) / 20
        sd = _std(window)
        if sd:
            f["bb_pctb_20"] = (c[-1] - (ma - 2 * sd)) / (4 * sd)
    if n >= 21:
        logs = [math.log(c[i] / c[i - 1]) for i in range(n - 20, n) if c[i - 1] > 0 and c[i] > 0]
        sd = _std(logs)
        f["realized_vol_20"] = None if sd is None else sd * math.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return f


def _has_ohlc(r: _Ohlc) -> bool:
    return r.open is not None and r.high is not None and r.low is not None


def _candle_features(rows: Sequence[_Ohlc]) -> dict[str, float | None]:
    """Candle shape features over the most recent bars.

    Windowed features (5-day body, ATR(14), 20-day range volatility) require
    the *consecutive* most recent bars to all have OHLC — a gap in the window
    yields ``None`` rather than silently mixing non-adjacent days.
    """
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["candle"])
    if not rows or not _has_ohlc(rows[-1]):
        return f
    last = rows[-1]
    o, h, low, c = last.open, last.high, last.low, last.close  # type: ignore[assignment]
    rng = h - low  # type: ignore[operator]
    if rng > 0:
        f["body_ratio"] = abs(c - o) / rng  # type: ignore[operator]
        f["upper_shadow_ratio"] = (h - max(o, c)) / rng  # type: ignore[operator,type-var]
        f["lower_shadow_ratio"] = (min(o, c) - low) / rng  # type: ignore[operator,type-var]
    if len(rows) >= 2 and rows[-2].close:
        f["gap_1d"] = _pct(o / rows[-2].close - 1.0)  # type: ignore[operator]
    if o:
        f["intraday_ret_1d"] = _pct(c / o - 1.0)
    last5 = rows[-5:]
    if len(last5) == 5 and all(_has_ohlc(r) and r.high > r.low for r in last5):  # type: ignore[operator]
        f["avg_body_ratio_5"] = sum(
            abs(r.close - r.open) / (r.high - r.low) for r in last5  # type: ignore[operator]
        ) / 5
    last15 = rows[-15:]
    if len(last15) == 15 and all(_has_ohlc(r) for r in last15[1:]):
        trs = [
            max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))  # type: ignore[operator]
            for prev, cur in zip(last15[:-1], last15[1:], strict=True)
        ]
        f["atr_14_pct"] = _pct(sum(trs) / 14 / c) if c else None
    last20 = rows[-20:]
    if len(last20) == 20 and all(
        _has_ohlc(r) and r.low > 0 and r.open > 0 for r in last20  # type: ignore[operator]
    ):
        hl = [math.log(r.high / r.low) ** 2 for r in last20]  # type: ignore[operator]
        f["parkinson_vol_20"] = (
            math.sqrt(sum(hl) / 20 / (4 * math.log(2)) * TRADING_DAYS_PER_YEAR) * 100
        )
        gk = [
            0.5 * math.log(r.high / r.low) ** 2  # type: ignore[operator]
            - (2 * math.log(2) - 1) * math.log(r.close / r.open) ** 2  # type: ignore[operator]
            for r in last20
        ]
        var = sum(gk) / 20
        f["garman_klass_vol_20"] = math.sqrt(max(var, 0.0) * TRADING_DAYS_PER_YEAR) * 100
    return f


def _liquidity_features(
    rows: Sequence[_Ohlc], shares_outstanding: float | None
) -> dict[str, float | None]:
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["liquidity"])
    vol_rows = [r for r in rows[-21:] if r.volume is not None]
    if len(rows) < 21 or len(vol_rows) < 21:
        return f
    value = [r.raw_close * r.volume for r in rows[-20:]]  # type: ignore[operator]
    mean20 = sum(value) / 20
    if mean20 > 0:
        f["log_traded_value_20d"] = math.log(mean20)
        f["traded_value_ratio_5_20"] = (sum(value[-5:]) / 5) / mean20
    rets = _returns([r.close for r in rows[-21:]])
    illiq = [abs(ret) / v for ret, v in zip(rets, value, strict=False) if v > 0]
    if illiq:
        f["amihud_illiq_20"] = sum(illiq) / len(illiq) * 1e9
    if shares_outstanding:
        mean_vol = sum(r.volume for r in rows[-20:]) / 20  # type: ignore[misc]
        f["share_turnover_20"] = _pct(mean_vol / shares_outstanding)
    return f


def _valuation_quality_features(
    statement: FinancialStatement | None, raw_price: float | None
) -> dict[str, float | None]:
    names = TRADING_FEATURE_GROUPS["valuation"] + TRADING_FEATURE_GROUPS["quality"]
    f: dict[str, float | None] = dict.fromkeys(names)
    if statement is None:
        return f
    shares = statement.shares_outstanding
    mcap = raw_price * shares if (raw_price and shares) else None
    if mcap and mcap > 0:
        f["log_market_cap"] = math.log(mcap)
        if statement.forecast_net_income is not None:
            f["forecast_earnings_yield"] = _pct(statement.forecast_net_income / mcap)
        elif statement.forecast_eps is not None and raw_price:
            f["forecast_earnings_yield"] = _pct(statement.forecast_eps / raw_price)
        if statement.operating_cash_flow is not None and statement.capital_expenditure is not None:
            fcf = statement.operating_cash_flow - abs(statement.capital_expenditure)
            f["fcf_yield"] = _pct(fcf / mcap)
    if statement.forecast_dividends_per_share is not None and raw_price:
        f["forecast_dividend_yield"] = _pct(statement.forecast_dividends_per_share / raw_price)
    if statement.total_assets:
        if statement.gross_profit is not None:
            f["gross_profitability"] = _pct(statement.gross_profit / statement.total_assets)
        if statement.net_income is not None and statement.operating_cash_flow is not None:
            f["accruals"] = _pct(
                (statement.net_income - statement.operating_cash_flow) / statement.total_assets
            )
    return f


def _guidance_target(statement: FinancialStatement) -> int | None:
    if statement.forecast_fiscal_year is not None:
        return statement.forecast_fiscal_year
    return statement.fiscal_year


def _event_features(
    *,
    latest: FinancialStatement | None,
    latest_public: date | None,
    earlier_disclosures: Sequence[FinancialStatement],
    rows: Sequence[_Ohlc],
    decision_date: date,
    earnings_dates: Sequence[date] | None,
) -> dict[str, float | None]:
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["events"])
    if latest is not None:
        target = _guidance_target(latest)
        # guidance revision vs the most recent EARLIER guidance for the same year
        if latest.forecast_eps is not None and target is not None:
            prior = [
                s for s in earlier_disclosures
                if s.forecast_eps is not None and _guidance_target(s) == target
            ]
            if prior and prior[-1].forecast_eps:
                old = prior[-1].forecast_eps
                f["forecast_eps_revision"] = _pct((latest.forecast_eps - old) / abs(old))
        # surprise: realised EPS vs the last guidance issued for that fiscal year
        if latest.eps is not None and latest.fiscal_year is not None:
            guided = [
                s for s in earlier_disclosures
                if s.forecast_eps is not None and _guidance_target(s) == latest.fiscal_year
            ]
            if guided and guided[-1].forecast_eps:
                g = guided[-1].forecast_eps
                f["earnings_surprise"] = _pct((latest.eps - g) / abs(g))
    if latest_public is not None:
        f["days_since_disclosure"] = float((decision_date - latest_public).days)
        before = [r for r in rows if r.date < latest_public]
        if before and before[-1].close and rows:
            f["post_disclosure_return"] = _pct(rows[-1].close / before[-1].close - 1.0)
    if earnings_dates:
        upcoming = [d for d in earnings_dates if d > decision_date]
        if upcoming:
            lead = (upcoming[0] - decision_date).days
            if lead <= MAX_KNOWN_EARNINGS_LEAD_DAYS:
                f["days_to_next_earnings"] = float(lead)
    return f


def _supply_features(
    margin: Sequence[MarginBalance] | None,
    rows: Sequence[_Ohlc],
    decision_date: date,
) -> dict[str, float | None]:
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["supply"])
    if not margin:
        return f
    public = [
        m for m in margin
        if (m.published_date or (m.date + timedelta(days=DEFAULT_MARGIN_PUBLICATION_LAG_DAYS)))
        <= decision_date
    ]
    if not public:
        return f
    latest = public[-1]
    if latest.long_balance is not None and latest.short_balance is not None:
        f["margin_ratio_log"] = math.log((latest.long_balance + 1.0) / (latest.short_balance + 1.0))
    vols = [r.volume for r in rows[-20:] if r.volume is not None]
    if latest.long_balance is not None and len(vols) == 20 and sum(vols) > 0:
        f["margin_long_to_volume"] = latest.long_balance / (sum(vols) / 20)
    return f


def _risk_features(
    rows: Sequence[_Ohlc], market_returns: Mapping[date, float] | None
) -> dict[str, float | None]:
    f: dict[str, float | None] = dict.fromkeys(TRADING_FEATURE_GROUPS["risk"])
    if not market_returns or len(rows) < 61:
        return f
    pairs: list[tuple[float, float]] = []
    for prev, cur in zip(rows[-61:-1], rows[-60:], strict=True):
        if prev.close and cur.date in market_returns:
            pairs.append((cur.close / prev.close - 1.0, market_returns[cur.date]))
    if len(pairs) < 40:
        return f
    xs = [m for _, m in pairs]
    ys = [s for s, _ in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return f
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var
    resid = [y - my - beta * (x - mx) for x, y in zip(xs, ys, strict=True)]
    sd = _std(resid)
    f["beta_60d"] = beta
    f["idio_vol_60d"] = None if sd is None else sd * math.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return f


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def equal_weight_market_returns(prices: Mapping[str, Sequence[PriceBar]]) -> dict[date, float]:
    """Daily equal-weighted universe return (market proxy), adjusted closes.

    Each day's value uses only that day's and the previous day's prices, so
    using it for dates <= a decision date introduces no look-ahead.
    """
    sums: dict[date, float] = {}
    counts: dict[date, int] = {}
    for bars in prices.values():
        rows = adjusted_ohlc(bars)
        for prev, cur in zip(rows, rows[1:], strict=False):
            if prev.close:
                sums[cur.date] = sums.get(cur.date, 0.0) + (cur.close / prev.close - 1.0)
                counts[cur.date] = counts.get(cur.date, 0) + 1
    return {d: sums[d] / counts[d] for d in sums}


def traded_value_20d(bars_as_of: Sequence[PriceBar]) -> float | None:
    """Mean 20-day traded value (unadjusted close x volume) — liquidity filter."""
    rows = sorted(bars_as_of, key=lambda b: b.date)[-20:]
    if len(rows) < 20 or any(b.volume is None for b in rows):
        return None
    return sum(float(b.close) * float(b.volume) for b in rows) / 20  # type: ignore[arg-type]


def compute_trading_features(
    *,
    statement: FinancialStatement | None,
    statement_public_date: date | None,
    earlier_disclosures: Sequence[FinancialStatement],
    bars_as_of: Sequence[PriceBar],
    decision_date: date,
    market_returns: Mapping[date, float] | None = None,
    earnings_dates: Sequence[date] | None = None,
    margin: Sequence[MarginBalance] | None = None,
    adjusted_rows: Sequence[_Ohlc] | None = None,
) -> dict[str, float | None]:
    """All extended features for one (ticker, decision date); ``None`` if unavailable.

    ``bars_as_of`` must already exclude bars after ``decision_date``;
    ``earlier_disclosures`` are statements public before ``statement`` (oldest
    first) and are used only for guidance revision / surprise.
    ``adjusted_rows`` (optional) are pre-computed ``adjusted_ohlc`` rows for the
    same window — the dataset builder passes them so the per-ticker adjustment
    runs once instead of once per decision date.
    """
    source = adjusted_rows if adjusted_rows is not None else adjusted_ohlc(bars_as_of)
    rows = [r for r in source if r.date <= decision_date]
    raw_price = rows[-1].raw_close if rows else None
    features: dict[str, float | None] = {}
    features.update(_price_features(rows))
    features.update(_candle_features(rows))
    features.update(
        _liquidity_features(rows, statement.shares_outstanding if statement else None)
    )
    features.update(_valuation_quality_features(statement, raw_price))
    features.update(
        _event_features(
            latest=statement,
            latest_public=statement_public_date,
            earlier_disclosures=earlier_disclosures,
            rows=rows,
            decision_date=decision_date,
            earnings_dates=earnings_dates,
        )
    )
    features.update(_supply_features(margin, rows, decision_date))
    features.update(_risk_features(rows, market_returns))
    return {name: features.get(name) for name in TRADING_FEATURES}


__all__ = [
    "DEFAULT_MARGIN_PUBLICATION_LAG_DAYS",
    "MAX_KNOWN_EARNINGS_LEAD_DAYS",
    "TRADING_FEATURES",
    "TRADING_FEATURE_GROUPS",
    "adjusted_ohlc",
    "compute_trading_features",
    "equal_weight_market_returns",
    "traded_value_20d",
]
