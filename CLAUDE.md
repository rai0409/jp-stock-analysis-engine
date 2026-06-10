# CLAUDE.md

## Project Goal

Build a production-quality Japanese stock analysis engine for public-release-quality code and self-use trading research.

This project is NOT:
- a paid SaaS product
- an investment advisory service for third parties
- a RAG service
- an EDINET ingestion platform clone
- a broker integration
- an automated trading bot

This project IS:
- a modular Japanese stock analysis engine
- a research-grade screening tool
- a self-use trading-signal research tool
- a reproducible Python package with tests and reports

## Required Modes

Always preserve these modes:

1. `analysis_only`
   - Default mode.
   - Outputs metrics, scores, risks, evidence, and reports.
   - Does not output buy/sell labels.

2. `screening`
   - Outputs candidate labels:
     - `strong_candidate`
     - `candidate`
     - `watchlist`
     - `avoid_candidate`
     - `insufficient_data`

3. `trade_signal`
   - Explicit opt-in only.
   - Outputs research signals:
     - `buy_signal`
     - `hold_signal`
     - `sell_signal`
     - `watch_signal`
     - `avoid_signal`
     - `insufficient_data`
   - Must include thresholds, evidence, confidence, rationale, blocking risks, and supporting factors.

## Coding Standards

- Prefer small typed modules over one large script.
- Use deterministic local fixtures for tests.
- Do not require network access in tests.
- Do not require paid APIs.
- Keep providers optional and stubbed unless explicitly requested.
- Do not add heavy ML frameworks unless requested.
- Avoid hidden state and global mutable configuration.
- Use clear warnings instead of crashing on missing financial data.
- Keep output reproducible.

## Financial Analysis Rules

- Never fabricate missing data.
- Always lower confidence when data is missing.
- Always include warnings for insufficient history.
- Always separate:
  - raw metrics
  - derived metrics
  - scores
  - screening labels
  - trade signals
- Valuation alone must never create a buy/sell signal.
- Trade signals must depend on multiple dimensions:
  - fundamentals
  - valuation
  - momentum
  - disclosure/NLP
  - risk flags
  - confidence

## Report Rules

Every report must include:

“This output is for analytical and self-directed research purposes. It is not personalized financial advice.”

Reports must include:
- data coverage
- metrics
- score breakdown
- risk flags
- evidence
- warnings
- limitations

## Testing Rules

Before final response, run:

- `python -m pytest`
- `ruff check .`

If tests cannot run, report the exact error. Do not claim success without running.

## Out of Scope

Do not implement in this repo:
- RAG service
- chatbot integration
- broker order execution
- auto-trading
- portfolio allocation advice
- position sizing
- margin/leverage/derivatives logic
- paid advisory workflows
