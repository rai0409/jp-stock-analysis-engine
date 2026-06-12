"""Per-ticker Markdown report writer.

The screening section appears only in ``screening``/``trade_signal`` modes,
and the research-signal section only in ``trade_signal`` mode. Every report
ends with the mandatory disclaimer.
"""

from __future__ import annotations

from pathlib import Path

from jp_stock_analysis.analysis.reliability import ReliabilityAssessment, assess_reliability
from jp_stock_analysis.config import AnalysisConfig
from jp_stock_analysis.schemas import StockAnalysisResult


def _fmt(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def _coverage_row(name: str, available: bool, note: str) -> str:
    return f"| {name} | {'yes' if available else 'no'} | {note} |"


def _executive_summary(
    result: StockAnalysisResult, assessment: ReliabilityAssessment
) -> list[str]:
    score = result.score
    lines = [
        f"- Ticker: `{result.ticker}`"
        + (f" — {result.company_name}" if result.company_name else ""),
        f"- Analysis date: {result.analysis_date.isoformat()}",
        f"- Mode: `{result.signal_mode}`",
        f"- Final score: {_fmt(score.final_score if score else None, digits=1)} / 100",
        f"- Confidence: {_fmt(score.confidence_score if score else None, digits=1)} / 100",
        f"- Risk score: {_fmt(score.risk_score if score else None, digits=1)} / 100",
        f"- Data coverage: {_fmt(assessment.data_coverage_score, digits=1)} / 100",
        "- Screening score (reliability-adjusted): "
        + f"{_fmt(assessment.screening_score, digits=1)} / 100",
        f"- Reliability grade: `{assessment.reliability_grade}`",
        f"- Screening eligible: {'yes' if assessment.screening_eligible else 'no'}",
    ]
    if assessment.reliability_grade == "low":
        final = score.final_score if score else None
        confidence = score.confidence_score if score else 0.0
        lines.append(
            f"- **Low reliability:** final score {_fmt(final, digits=1)} is computed from "
            f"limited data (coverage {assessment.data_coverage_score:.1f}/100, confidence "
            f"{confidence:.1f}/100) and must NOT be read as a strong candidate."
        )
    if result.signal_mode == "screening" and result.screening_label:
        lines.append(f"- Screening label: `{result.screening_label}`")
    if result.signal_mode == "trade_signal" and result.signal:
        lines.append(f"- Research signal: `{result.signal.label}`")
    return lines


def _data_coverage(result: StockAnalysisResult) -> list[str]:
    fundamentals = result.fundamentals
    momentum = result.momentum
    rows = [
        "| Component | Available | Notes |",
        "|---|---|---|",
        _coverage_row(
            "Fundamentals",
            fundamentals is not None,
            f"fiscal year {fundamentals.fiscal_year}" if fundamentals else "no statements",
        ),
        _coverage_row(
            "Valuation",
            result.valuation is not None,
            f"classification: {result.valuation.valuation_classification}"
            if result.valuation
            else "no market price or statement",
        ),
        _coverage_row(
            "Momentum",
            momentum is not None,
            f"{momentum.observations} daily bars" if momentum else "no price history",
        ),
        _coverage_row(
            "Disclosure",
            result.disclosure is not None,
            f"{len(result.disclosure.findings)} findings"
            if result.disclosure
            else "no disclosure text",
        ),
        _coverage_row(
            "Risk",
            result.risks is not None,
            f"{len(result.risks.flags)} flags" if result.risks else "not assessed",
        ),
    ]
    return rows


def _fundamentals_section(result: StockAnalysisResult) -> list[str]:
    f = result.fundamentals
    if f is None:
        return ["No fundamentals data available."]
    return [
        "| Metric | Value |",
        "|---|---|",
        f"| Revenue growth YoY | {_fmt(f.revenue_growth_yoy, '%')} |",
        f"| Operating income growth YoY | {_fmt(f.operating_income_growth_yoy, '%')} |",
        f"| Net income growth YoY | {_fmt(f.net_income_growth_yoy, '%')} |",
        f"| EPS growth YoY | {_fmt(f.eps_growth_yoy, '%')} |",
        f"| Operating margin | {_fmt(f.operating_margin, '%')} |",
        f"| Net margin | {_fmt(f.net_margin, '%')} |",
        f"| ROE | {_fmt(f.roe, '%')} |",
        f"| ROA | {_fmt(f.roa, '%')} |",
        f"| Equity ratio | {_fmt(f.equity_ratio, '%')} |",
        f"| FCF margin | {_fmt(f.fcf_margin, '%')} |",
        f"| Dividend payout ratio | {_fmt(f.dividend_payout_ratio, '%')} |",
    ]


def _valuation_section(result: StockAnalysisResult) -> list[str]:
    v = result.valuation
    if v is None:
        return ["No valuation data available."]
    return [
        "| Metric | Value |",
        "|---|---|",
        f"| Market price | {_fmt(v.market_price)} |",
        f"| PER | {_fmt(v.per)} |",
        f"| PBR | {_fmt(v.pbr)} |",
        f"| PSR | {_fmt(v.psr)} |",
        f"| Dividend yield | {_fmt(v.dividend_yield, '%')} |",
        f"| PEG | {_fmt(v.peg)} |",
        f"| Market cap | {_fmt(v.market_cap, digits=0)} |",
        f"| Classification | {v.valuation_classification} |",
    ]


def _momentum_section(result: StockAnalysisResult) -> list[str]:
    m = result.momentum
    if m is None:
        return ["No price history available."]
    return [
        "| Metric | Value |",
        "|---|---|",
        f"| Return 1M | {_fmt(m.return_1m, '%')} |",
        f"| Return 3M | {_fmt(m.return_3m, '%')} |",
        f"| Return 6M | {_fmt(m.return_6m, '%')} |",
        f"| Return 12M | {_fmt(m.return_12m, '%')} |",
        f"| MA 20d | {_fmt(m.moving_average_20d)} |",
        f"| MA 60d | {_fmt(m.moving_average_60d)} |",
        f"| MA 120d | {_fmt(m.moving_average_120d)} |",
        f"| MA 200d | {_fmt(m.moving_average_200d)} |",
        f"| Volatility (annualized) | {_fmt(m.volatility_annualized, '%')} |",
        f"| Max drawdown | {_fmt(m.max_drawdown, '%')} |",
        f"| Volume trend | {m.volume_trend or 'n/a'} |",
    ]


def _disclosure_section(result: StockAnalysisResult) -> list[str]:
    d = result.disclosure
    if d is None:
        return ["No disclosure text available."]
    lines = [
        f"Analyzer: `{d.analyzer}` — tone score {d.tone_score:.0f} "
        f"(positive: {d.positive_count}, negative: {d.negative_count}, "
        f"risk: {d.risk_count}, uncertainty: {d.uncertainty_count})",
        "",
    ]
    if d.findings:
        lines.extend(["| Rule | Category | Severity | Evidence |", "|---|---|---|---|"])
        for finding in d.findings:
            evidence = finding.evidence_text.replace("|", "\\|")
            lines.append(
                f"| {finding.rule_id} | {finding.category} | {finding.severity} | {evidence} |"
            )
    else:
        lines.append("No findings extracted.")
    return lines


def _risk_section(result: StockAnalysisResult) -> list[str]:
    r = result.risks
    if r is None:
        return ["Risk analysis not performed."]
    lines = [f"Risk score: {r.risk_score:.1f} / 100 (0 = low risk)", ""]
    if r.flags:
        lines.extend(["| Flag | Severity | Explanation |", "|---|---|---|"])
        for flag in r.flags:
            lines.append(f"| {flag.risk_id} | {flag.severity} | {flag.explanation} |")
    else:
        lines.append("No risk flags raised.")
    return lines


def _score_section(result: StockAnalysisResult) -> list[str]:
    s = result.score
    if s is None:
        return ["No score computed."]
    lines = [
        "| Sub-score | Value | Reason |",
        "|---|---|---|",
    ]
    for name in (
        "quality_score",
        "growth_score",
        "valuation_score",
        "momentum_score",
        "disclosure_score",
        "risk_score",
        "final_score",
    ):
        value = getattr(s, name)
        reason = s.reasons.get(name, "").replace("|", "\\|")
        lines.append(f"| {name} | {_fmt(value, digits=1)} | {reason} |")
    lines.append("")
    lines.append(f"Confidence score: {s.confidence_score:.1f} / 100")
    return lines


def _sector_relative_section(result: StockAnalysisResult) -> list[str]:
    s = result.sector_relative
    if s is None:
        return ["No sector-relative data available."]
    lines = [
        f"Sector: {s.sector} — {s.peer_count} same-sector companies in this universe. "
        "100 = most favorable within the sector. "
        "Sector-relative score is reported separately from the final score.",
        "",
        "| Metric | Percentile |",
        "|---|---|",
        f"| PER (cheaper = higher) | {_fmt(s.per_percentile, digits=1)} |",
        f"| PBR (cheaper = higher) | {_fmt(s.pbr_percentile, digits=1)} |",
        f"| Revenue growth | {_fmt(s.revenue_growth_percentile, digits=1)} |",
        f"| Operating margin | {_fmt(s.operating_margin_percentile, digits=1)} |",
        f"| ROE | {_fmt(s.roe_percentile, digits=1)} |",
        f"| Momentum (3m, else 6m) | {_fmt(s.momentum_percentile, digits=1)} |",
        f"| Risk score (lower = higher) | {_fmt(s.risk_percentile, digits=1)} |",
        f"| **Sector-relative score** | {_fmt(s.sector_relative_score, digits=1)} |",
    ]
    return lines


def _signal_section(result: StockAnalysisResult) -> list[str]:
    signal = result.signal
    if signal is None:
        return ["No signal generated."]
    lines = [
        f"- Label: `{signal.label}`",
        f"- Confidence: {signal.confidence:.1f}",
        f"- Rationale: {signal.rationale}",
        "- Thresholds used: "
        + ", ".join(f"{k}={v:g}" for k, v in signal.thresholds_used.items()),
    ]
    if signal.supporting_factors:
        lines.append("- Supporting factors:")
        lines.extend(f"  - {factor}" for factor in signal.supporting_factors)
    if signal.blocking_risks:
        lines.append("- Blocking risks:")
        lines.extend(f"  - {risk}" for risk in signal.blocking_risks)
    lines.extend(["", f"> {signal.disclaimer}"])
    return lines


def _evidence_and_warnings(
    result: StockAnalysisResult, assessment: ReliabilityAssessment
) -> list[str]:
    lines: list[str] = []
    evidence: list[str] = []
    if result.disclosure:
        evidence.extend(
            f"[{finding.rule_id}] {finding.evidence_text}" for finding in result.disclosure.findings
        )
    if result.risks:
        for flag in result.risks.flags:
            evidence.extend(f"[{flag.risk_id}] {item}" for item in flag.evidence)
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("Evidence: none collected.")
    lines.append("")

    warnings: list[str] = list(result.warnings)
    for name, component in (
        ("fundamentals", result.fundamentals),
        ("valuation", result.valuation),
        ("momentum", result.momentum),
        ("disclosure", result.disclosure),
        ("risk", result.risks),
        ("score", result.score),
    ):
        if component is not None:
            warnings.extend(f"{name}: {warning}" for warning in component.warnings)
    warnings.extend(assessment.warnings)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none.")
    return lines


def _limitations(result: StockAnalysisResult) -> list[str]:
    lines = [
        "- Deterministic rule-based analysis of local data only; no real-time feeds.",
        "- Disclosure analysis uses keyword matching, not full language understanding.",
        "- Scores are research heuristics, not forecasts of future returns.",
        "- No position sizing, portfolio allocation, or execution guidance is provided.",
    ]
    if result.momentum is not None and result.momentum.observations < 252:
        lines.append(
            f"- Price history covers only {result.momentum.observations} trading days; "
            "long-horizon momentum metrics may be unavailable."
        )
    if result.fundamentals is None:
        lines.append("- No financial statements were provided for this ticker.")
    if result.disclosure is None:
        lines.append("- No disclosure text was provided for this ticker.")
    return lines


def render_markdown_report(result: StockAnalysisResult, config: AnalysisConfig) -> str:
    """Render the full per-ticker Markdown report."""
    title = result.ticker + (f" — {result.company_name}" if result.company_name else "")
    assessment = assess_reliability(result, config.thresholds)
    sections: list[tuple[str, list[str]]] = [
        ("Executive Summary", _executive_summary(result, assessment)),
        ("Data Coverage", _data_coverage(result)),
        ("Fundamental Metrics", _fundamentals_section(result)),
        ("Valuation Metrics", _valuation_section(result)),
        ("Momentum Metrics", _momentum_section(result)),
        ("Disclosure Analysis", _disclosure_section(result)),
        ("Risk Flags", _risk_section(result)),
        ("Integrated Score", _score_section(result)),
    ]
    if result.sector_relative is not None:
        sections.append(("Sector Relative", _sector_relative_section(result)))
    if result.signal_mode in ("screening", "trade_signal"):
        label = result.screening_label or "not assigned"
        sections.append(("Screening", [f"Screening label: `{label}`"]))
    if result.signal_mode == "trade_signal":
        sections.append(("Research Signal", _signal_section(result)))
    sections.extend(
        [
            ("Evidence and Warnings", _evidence_and_warnings(result, assessment)),
            ("Limitations", _limitations(result)),
            ("Disclaimer", [config.disclaimer]),
        ]
    )

    lines = [f"# Stock Analysis Report: {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.extend(body)
        lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    result: StockAnalysisResult, output_dir: str | Path, config: AnalysisConfig
) -> Path:
    """Write ``<ticker>.md`` into the output directory and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.ticker}.md"
    path.write_text(render_markdown_report(result, config), encoding="utf-8")
    return path
