"""Run the offline commercial-quality J-Quants analysis improvement pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jp_stock_analysis.analysis.jquants_abnormal_return import (
    DEFAULT_OUTPUT as ABNORMAL_OUTPUT,
    DEFAULT_SUMMARY as ABNORMAL_SUMMARY,
    build_abnormal_return,
)
from jp_stock_analysis.analysis.jquants_commercial_common import (
    ANALYSIS_DIR,
    DEFAULT_CANDIDATES,
    DEFAULT_PRICES,
    LOGIC_VERSION,
    jst_now_iso,
    write_json,
)
from jp_stock_analysis.analysis.jquants_liquidity import (
    DEFAULT_OUTPUT as LIQUIDITY_OUTPUT,
    DEFAULT_SUMMARY as LIQUIDITY_SUMMARY,
    build_liquidity_features,
)
from jp_stock_analysis.analysis.jquants_price_reaction_v2 import (
    DEFAULT_OUTPUT as REACTION_OUTPUT,
    DEFAULT_SUMMARY as REACTION_SUMMARY,
    build_price_reaction_v2,
)
from jp_stock_analysis.analysis.jquants_commercial_quality_rankings import (
    DEFAULT_EXCLUDED_AUDIT as RANKINGS_EXCLUDED_AUDIT,
    DEFAULT_NEGATIVE_REACTED as NEGATIVE_REACTED_RANKING,
    DEFAULT_NEGATIVE_REVERSAL as NEGATIVE_REVERSAL_RANKING,
    DEFAULT_POSITIVE_REACTED as POSITIVE_REACTED_RANKING,
    DEFAULT_POSITIVE_UNREACTED as POSITIVE_UNREACTED_RANKING,
    DEFAULT_SUMMARY as RANKINGS_SUMMARY,
    DEFAULT_UNCLASSIFIED_AUDIT as RANKINGS_UNCLASSIFIED_AUDIT,
    build_commercial_quality_rankings,
)
from jp_stock_analysis.analysis.jquants_commercial_quality_enrichment import (
    DEFAULT_SUMMARY as ENRICHED_RANKINGS_SUMMARY,
    build_commercial_quality_enriched_rankings,
)
from jp_stock_analysis.analysis.jquants_robust_summary import (
    DEFAULT_OUTPUT as ROBUST_OUTPUT,
    DEFAULT_SUMMARY as ROBUST_SUMMARY,
    build_robust_summary,
)


DEFAULT_FINAL_SUMMARY = ANALYSIS_DIR / "commercial_quality_improvement_final_summary.json"


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_commercial_quality_dataset(
    candidates_path: Path = DEFAULT_CANDIDATES,
    prices_path: Path = DEFAULT_PRICES,
    final_summary_path: Path = DEFAULT_FINAL_SUMMARY,
) -> dict[str, object]:
    reaction_summary = build_price_reaction_v2(
        candidates_path, prices_path, REACTION_OUTPUT, REACTION_SUMMARY
    )
    abnormal_summary = build_abnormal_return(
        REACTION_OUTPUT, prices_path, ABNORMAL_OUTPUT, ABNORMAL_SUMMARY
    )
    liquidity_summary = build_liquidity_features(
        ABNORMAL_OUTPUT, prices_path, LIQUIDITY_OUTPUT, LIQUIDITY_SUMMARY
    )
    robust_summary = build_robust_summary(LIQUIDITY_OUTPUT, ROBUST_OUTPUT, ROBUST_SUMMARY)
    rankings_summary = build_commercial_quality_rankings(LIQUIDITY_OUTPUT)
    enriched_summary = build_commercial_quality_enriched_rankings()

    warnings: list[str] = []
    no_price = reaction_summary.get("price_reaction_status_v2_counts", {}).get(
        "no_price_after_event", 0
    )
    if no_price:
        warnings.append(f"{no_price} rows have no price after event")
    invalid_abnormal = _load_json(ABNORMAL_SUMMARY).get("abnormal_ret_status_counts", {})
    if invalid_abnormal:
        invalid_1d = invalid_abnormal.get("abnormal_ret_1d_status", {}).get(
            "invalid_source_return", 0
        )
        if invalid_1d:
            warnings.append(f"{invalid_1d} rows have invalid 1d source return")
    warnings.extend(str(w) for w in rankings_summary.get("warnings", []))
    warnings.extend(str(w) for w in enriched_summary.get("warnings", []))

    final = {
        "logic_version": LOGIC_VERSION,
        "generated_at_jst": jst_now_iso(),
        "output_paths": {
            "price_reaction_v2_csv": str(REACTION_OUTPUT),
            "price_reaction_v2_summary_json": str(REACTION_SUMMARY),
            "abnormal_return_csv": str(ABNORMAL_OUTPUT),
            "abnormal_return_summary_json": str(ABNORMAL_SUMMARY),
            "liquidity_csv": str(LIQUIDITY_OUTPUT),
            "liquidity_summary_json": str(LIQUIDITY_SUMMARY),
            "robust_summary_csv": str(ROBUST_OUTPUT),
            "robust_summary_summary_json": str(ROBUST_SUMMARY),
            "positive_reacted_ranking_csv": str(POSITIVE_REACTED_RANKING),
            "positive_unreacted_ranking_csv": str(POSITIVE_UNREACTED_RANKING),
            "negative_reacted_ranking_csv": str(NEGATIVE_REACTED_RANKING),
            "negative_reversal_ranking_csv": str(NEGATIVE_REVERSAL_RANKING),
            "rankings_excluded_audit_csv": str(RANKINGS_EXCLUDED_AUDIT),
            "rankings_unclassified_audit_csv": str(RANKINGS_UNCLASSIFIED_AUDIT),
            "rankings_summary_json": str(RANKINGS_SUMMARY),
            "enriched_rankings_summary_json": str(ENRICHED_RANKINGS_SUMMARY),
            "final_summary_json": str(final_summary_path),
        },
        "input_row_counts": {
            "commercial_revision_candidates": reaction_summary.get("input_rows"),
            "price_reaction_v2": abnormal_summary.get("input_rows"),
            "abnormal_return": liquidity_summary.get("input_rows"),
            "liquidity": robust_summary.get("rows"),
            "rankings": rankings_summary.get("input_rows"),
        },
        "output_row_counts": {
            "price_reaction_v2": reaction_summary.get("output_rows"),
            "abnormal_return": abnormal_summary.get("output_rows"),
            "liquidity": liquidity_summary.get("output_rows"),
            "robust_summary": robust_summary.get("groups"),
            "rankings_eligible": rankings_summary.get("eligible_rows"),
            "rankings_excluded": rankings_summary.get("excluded_rows"),
            "rankings_unclassified": rankings_summary.get("unclassified_rows"),
            "enriched_rankings": enriched_summary.get("total_rows"),
        },
        "reaction_v2_status_counts": reaction_summary.get("price_reaction_status_v2_counts", {}),
        "abnormal_return_status_counts": abnormal_summary.get("abnormal_ret_status_counts", {}),
        "liquidity_bucket_counts": liquidity_summary.get("liquidity_bucket_counts", {}),
        "robust_summary_groups": robust_summary.get("groups"),
        "rankings_counts": rankings_summary.get("ranking_counts", {}),
        "rankings_output_paths": rankings_summary.get("output_paths", {}),
        "rankings": {
            "ranking_counts": rankings_summary.get("ranking_counts", {}),
            "eligible_rows": rankings_summary.get("eligible_rows"),
            "excluded_rows": rankings_summary.get("excluded_rows"),
            "unclassified_rows": rankings_summary.get("unclassified_rows"),
            "unclassified_output_path": rankings_summary.get("unclassified_output_path"),
            "unclassified_reason_counts": rankings_summary.get(
                "unclassified_reason_counts", {}
            ),
            "output_paths": rankings_summary.get("output_paths", {}),
        },
        "enriched_rankings": {
            "output_files": enriched_summary.get("output_files", {}),
            "total_rows": enriched_summary.get("total_rows"),
            "total_metadata_missing_rows": enriched_summary.get(
                "total_metadata_missing_rows"
            ),
            "metadata_input_path": enriched_summary.get("metadata_input_path"),
            "warnings": enriched_summary.get("warnings", []),
        },
        "warnings": warnings,
    }
    write_json(final_summary_path, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--final-summary", type=Path, default=DEFAULT_FINAL_SUMMARY)
    args = parser.parse_args(argv)
    build_commercial_quality_dataset(args.candidates, args.prices, args.final_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
