# Data Contract

## Scope

This document fixes the file-level contract between
`topix1000_disclosure_platform` and `jp-stock-analysis-engine` before a future
gold export/import connection is implemented.

Responsibilities are separated as follows:

- `topix1000_disclosure_platform` owns ingestion and export. It is the
  raw/bronze/silver/gold export side for EDINET disclosures, security metadata,
  price-derived records, and curated gold datasets.
- `jp-stock-analysis-engine` owns import, analysis, validation, and dashboard
  presentation. It consumes exported records and produces research and
  validation support outputs.

This repository must not call upstream production APIs as part of this contract
and must not require direct access to the disclosure platform database.

## Point-in-Time Principle

Every imported record used for analysis must carry explicit point-in-time
metadata. Analysis may only use information that was available at or before the
decision timestamp being simulated or reviewed.

The minimum required point-in-time fields are:

- `available_from_timestamp`: earliest timestamp when the record could be used
  by this engine.
- `asof_timestamp`: timestamp of the analytical view being produced.
- `feature_cutoff_timestamp`: latest timestamp allowed for feature generation.
- `label_start_timestamp` and `label_end_timestamp`: future outcome window used
  only for validation labels, never for feature construction.

Features and signals must be reproducible from records whose
`available_from_timestamp` is less than or equal to the relevant cutoff.

## Future Leakage Rules

The following conditions are prohibited:

- Using a disclosure before its `disclosed_at` and `available_from_timestamp`.
- Using revised or corrected disclosure content without preserving
  `revision_of`, `content_hash`, and the new availability timestamp.
- Building features from prices, fundamentals, disclosures, security master
  attributes, or labels that were not available by `feature_cutoff_timestamp`.
- Computing a signal with label-window information in any feature input.
- Treating end-of-day price bars as available before their
  `available_from_timestamp`.
- Joining current security master attributes into historical analysis without
  respecting `effective_from` and `effective_to`.

Records that cannot prove point-in-time safety must set
`leakage_safe_flag` to `false` and `leakage_status` or `validation_status` to a
non-passing value. Such records are allowed for diagnostics only and require
analyst review.

## Schema Versioning

All JSON Schemas in `schemas/` use JSON Schema draft 2020-12 and include a
required `schema_version` field.

Versioning policy:

- Patch versions may clarify descriptions or add optional fields.
- Minor versions may add optional fields or new enum values when older
  consumers can ignore them safely.
- Major versions are required for renaming fields, changing required fields,
  changing meaning, or removing fields.
- Producers and consumers must reject unknown major versions unless an explicit
  compatibility adapter exists.

Schema files are source-controlled. Generated data, local exports, secrets, API
keys, `.env` files, local databases, and credentials are not source-controlled.

## Research-Only Boundary

Outputs from this repository are for research only, validation support, and
analyst review. They are not investment advice, trade recommendations, broker
instructions, automated order management, or execution-management integration.

Any downstream workflow that uses these records must keep
`research_only_flag` available on signal outputs and must require human analyst
review before any external use.

## Contract Files

The initial contract is split into five schemas:

- `schemas/disclosure_event.schema.json`
- `schemas/price_bar.schema.json`
- `schemas/security_master.schema.json`
- `schemas/point_in_time_feature.schema.json`
- `schemas/signal_output.schema.json`

These schemas define the minimum fields expected from future gold exports and
the minimum fields this engine must preserve when producing features and
research-only signal outputs.

## Gold Export Import Adapter

`Topix1000GoldExportProvider`
(`src/jp_stock_analysis/providers/topix1000_gold_export.py`) reads local gold
export files from a directory and returns pandas DataFrames for research and
validation support. It accepts JSONL files and JSON array files, validates the
repository-owned schema `required` fields, and checks point-in-time controls
before data is exposed to later analysis code.

The current adapter is a connection-preparation layer only. It uses synthetic
fixtures in tests and does not implement real data export, direct database
access, API access, credential handling, or changes in
`topix1000_disclosure_platform`.
