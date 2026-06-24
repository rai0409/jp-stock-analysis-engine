# J-Quants Listed Master Export

## Purpose

`fetch-jquants-listed-master` exports the current J-Quants listed master metadata
for a fixed universe, such as the TOPIX1000 ticker list. It is meant to help
diagnose universe tickers that are missing from a local J-Quants price store and
to enrich downstream analysis with company, sector, and market labels.

The command reuses the existing `JQuantsProvider` request path. Live calls use
the configured J-Quants V2 listed master endpoint and the `x-api-key` header;
the command does not add Authorization Bearer logic.

## Command Usage

```bash
.venv/bin/python -m jp_stock_analysis.cli fetch-jquants-listed-master \
  --universe-file /tmp/topix1000_tickers.csv \
  --output-file /tmp/jquants_topix1000_price_store/topix1000_listed_master_snapshot.csv \
  --report-file /tmp/jquants_topix1000_price_store/topix1000_listed_master_snapshot_report.json \
  --sleep-seconds 0.5 \
  --allow-network
```

`--allow-network` is required for the live metadata export. The command only
prechecks `JQUANTS_API_KEY` as `PRESENT` or `MISSING` in the report and never
prints or writes the key value.

## Output Files

The CSV writes exactly one row per universe ticker. Required columns are:

- `ticker`
- `name_universe`
- `universe_date`
- `new_index_category`
- `matched`
- `company_name`
- `sector`
- `market`
- `source_metadata_json`
- `error`

Convenience columns extracted from `source_metadata_json` are:

- `raw_code`
- `company_name_en`
- `sector_17`
- `sector_33`

The report JSON includes universe and match counts, missing tickers, counts by
`new_index_category`, the resolved listed master endpoint URL, API key status,
and `secret_included: false`.

## Missing Price Investigation

Compare `missing_tickers` from the listed master report with the ticker coverage
in the local price store. A ticker that matches listed master metadata but has no
price rows may be a newly listed name, a date-window issue, an endpoint coverage
issue, or a code normalization issue. A ticker that does not match listed master
metadata should be checked against the source universe file for stale, delisted,
or malformed codes. Alphanumeric tickers such as `167A`, `268A`, `285A`, `417A`,
`543A`, and `547A` are preserved as strings.

## CompanyMetadata Mapping

The existing `CompanyMetadata` schema is unchanged:

- `ticker`
- `company_name`
- `sector`
- `market`
- `source_metadata`

J-Quants listed master fields map as follows:

- `CoName`, `CoNameEn`, `CompanyName`, or `CompanyNameEnglish` -> `company_name`
- `S33Nm`, `S17Nm`, `Sector33CodeName`, or `Sector17CodeName` -> `sector`
- `MktNm` or `MarketCodeName` -> `market`
- `source_metadata.source` -> `jquants`
- `source_metadata.raw_code` -> `Code`, `LocalCode`, `code`, or `local_code`
- `source_metadata.company_name_en` -> `CoNameEn` or `CompanyNameEnglish`
- `source_metadata.sector_17` -> `S17Nm` or `Sector17CodeName`
- `source_metadata.sector_33` -> `S33Nm` or `Sector33CodeName`
- `source_metadata.market` -> `MktNm` or `MarketCodeName`

Missing values are omitted from `source_metadata`; they are not fabricated.

## No-Look-Ahead Caution

The listed master endpoint represents current listed metadata, not historical
point-in-time metadata. Use it for coverage diagnostics and descriptive labels,
but do not treat current sector, market, or company metadata as historical facts
for past model training or point-in-time backtests without a separate
point-in-time source.
