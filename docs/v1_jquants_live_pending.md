# J-Quants Live Validation Pending

Status:
- local provider: validated
- jquants-cache provider: validated
- jquants-live provider: implemented but real API validation is pending

Observed live API result:
- HTTP 403 Forbidden
- Example path:
  https://api.jquants.com/v2/prices/daily_quotes?code=7203&from=2025-06-11&to=2026-03-19

Decision:
- Live API validation is intentionally deferred.
- The project will continue using local fixtures and jquants-cache mode.
- No .cache/jquants files or API keys are committed.
- Tests remain fully offline.
- Future work should verify the official J-Quants endpoint, authentication method, plan permission, and date range.

Current safe modes:
- local
- jquants-cache

Current non-blocking limitation:
- jquants-live may require endpoint/auth/plan correction before production use.
