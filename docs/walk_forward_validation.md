# Walk-Forward & Purged/Embargo Validation (research-only)

**Scope:** `modeling/walk_forward.py`, `modeling/purged.py`.

> This output is for analytical and self-directed research purposes. It is not
> personalized financial advice.

## Walk-forward folds

`generate_folds` / `build_walk_forward_plan` split an ordered list of
decision-date **periods** into train/test folds:

- **expanding** — the training set grows to include all prior periods;
- **rolling** — the training set is the last `train_periods` periods;
- `min_train_periods` — minimum training history before the first test fold;
- `test_periods` — test-window size per fold (non-overlapping);
- horizon-aware — the plan records the horizons so a purge/embargo can drop
  training labels whose forward-return window overlaps the test period.

No look-ahead: a fold's training periods are always strictly before its test
periods (`train_end < test_start`). Folds are deterministic and emitted as a
stable fold table (JSON + Markdown). This is a domain-aware analogue of a
`TimeSeriesSplit`, specialised for disclosure dates and forward-return labels.

Example (5 monthly periods, expanding, `min_train_periods=2`, `test_periods=1`):

| fold | train | test |
| --- | --- | --- |
| 0 | m1..m2 | m3 |
| 1 | m1..m3 | m4 |
| 2 | m1..m4 | m5 |

## Purged / embargo splitting

`purge_embargo_split` removes leakage between a training set and a test window,
with no `mlfinlab` dependency:

- **purge** — drop training samples whose label window
  `[label_start, label_end]` overlaps the test window `[test_start, test_end]`.
  Touching endpoints count as overlap (conservative).
- **embargo** — additionally drop training samples that *start* within
  `embargo_days` after the test window ends, guarding against serial correlation
  immediately after the test set.

Each `LabeledSample` carries its forward-return label window; the
`ModelingObservation.label_window_end(bars, horizon)` helper derives the
realisation date from prices. The result reports kept samples and a per-reason
drop count (`purged_overlap`, `embargo_after_test`).

### Edge cases covered by tests

- no purge when the label window ends before the test window;
- overlapping training samples removed;
- label window touching `test_start` counts as overlap;
- embargo removes samples starting just after the test window, keeps those
  beyond the embargo horizon;
- invalid windows (`end < start`, negative embargo) raise.
