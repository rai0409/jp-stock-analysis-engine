"""Tests for the adopted primary tree model and the research transformer switch.

- primary model = purged walk-forward OOS tree ensemble (LightGBM + CatBoost +
  Ridge), explicit baseline fallback when no tree backend is available;
- research transformers are OFF by default, never import/require torch when
  OFF, and never change the primary model's outputs when ON.

Synthetic, deterministic, offline.
"""

from __future__ import annotations

import json

import pytest

from jp_stock_analysis.modeling import ml_models
from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.modeling.ml_models import (
    MODEL_BASELINE,
    MODEL_TREE_ENSEMBLE,
    train_primary_model,
)
from jp_stock_analysis.modeling.pipeline import PipelineConfig, run_pipeline
from jp_stock_analysis.modeling.research_transformers import (
    MODEL_CS_TRANSFORMER,
    MODEL_GRU,
    MODEL_TFT,
    RESEARCH_MODELS,
    STATUS_DISABLED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_MISSING_DEPENDENCY,
    STATUS_TRAINED,
    TransformerResearchConfig,
    _InputBuilder,
    run_transformer_research,
    torch_available,
)

STAMP = "1970-01-01T00:00:00Z"
TREES = ml_models.backend_available("lightgbm") and ml_models.backend_available("catboost")
needs_trees = pytest.mark.skipif(not TREES, reason="lightgbm/catboost not installed")
needs_torch = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def _bundle_and_dataset():
    b = build_synthetic_bundle()
    ds = build_modeling_dataset(
        b.fundamentals, b.prices, b.metadata, b.narratives,
        decision_dates=b.decision_dates, horizons=b.horizons,
        bundle_disclosure_date=b.bundle_disclosure_date, is_synthetic=True,
    )
    return b, ds


# --------------------------------------------------------------------------- #
# primary tree model
# --------------------------------------------------------------------------- #
@needs_trees
def test_primary_default_is_oos_tree_ensemble():
    b, ds = _bundle_and_dataset()
    primary = train_primary_model(ds, horizon=5, prices=b.prices)
    assert primary.model_label == MODEL_TREE_ENSEMBLE
    assert primary.fallback_used is False
    assert {"lightgbm_regressor", "catboost_regressor", "ridge"} <= set(primary.members)
    # OOS: the first decision date is training-only, never scored
    assert min(ds.decision_dates) not in {s.decision_date for s in primary.scored}


def test_primary_falls_back_to_baseline_explicitly(monkeypatch):
    _b, ds = _bundle_and_dataset()
    monkeypatch.setattr(ml_models, "backend_available", lambda name: False)
    primary = train_primary_model(ds, horizon=5)
    assert primary.model_label == MODEL_BASELINE
    assert primary.fallback_used is True
    assert any("falling back" in w for w in primary.warnings)


def test_primary_rejects_unknown_model():
    _b, ds = _bundle_and_dataset()
    with pytest.raises(ValueError):
        train_primary_model(ds, horizon=5, model="not_a_model")


@needs_trees
def test_pipeline_downstream_steps_use_primary_model(tmp_path):
    b, ds = _bundle_and_dataset()
    summary = run_pipeline(ds, b.prices, output_dir=tmp_path, run_id="r", fixed_timestamp=STAMP)
    run_dir = tmp_path / "r"
    assert summary["primary_model"] == MODEL_TREE_ENSEMBLE
    ranking = json.loads((run_dir / "ranking/ranking_metrics.json").read_text(encoding="utf-8"))
    assert ranking["model_label"] == MODEL_TREE_ENSEMBLE
    primary = json.loads((run_dir / "primary/primary_model.json").read_text(encoding="utf-8"))
    assert primary["validation"] == "walk_forward_purged_oos"
    report = json.loads((run_dir / "modeling_report.json").read_text(encoding="utf-8"))
    assert report["primary_model"]["model_label"] == MODEL_TREE_ENSEMBLE


def test_pipeline_legacy_baseline_primary_still_available(tmp_path):
    b, ds = _bundle_and_dataset()
    summary = run_pipeline(
        ds, b.prices, output_dir=tmp_path, run_id="r", fixed_timestamp=STAMP,
        config=PipelineConfig(primary_model=MODEL_BASELINE),
    )
    assert summary["primary_model"] == MODEL_BASELINE
    assert summary["primary_model_fallback_used"] is False


# --------------------------------------------------------------------------- #
# research transformer switch
# --------------------------------------------------------------------------- #
def test_research_transformers_off_by_default():
    _b, ds = _bundle_and_dataset()
    results = run_transformer_research(ds, horizon=5)
    assert [r.status for r in results] == [STATUS_DISABLED] * len(RESEARCH_MODELS)
    assert all(r.scored == [] for r in results)


def test_research_transformers_report_missing_torch(monkeypatch):
    from jp_stock_analysis.modeling import research_transformers as rt

    _b, ds = _bundle_and_dataset()
    monkeypatch.setattr(rt, "torch_available", lambda: False)
    results = rt.run_transformer_research(
        ds, horizon=5, config=TransformerResearchConfig(enabled=True)
    )
    assert {r.status for r in results} == {STATUS_MISSING_DEPENDENCY}


def test_research_config_validation():
    with pytest.raises(ValueError):
        TransformerResearchConfig(models=("not_a_model",))
    with pytest.raises(ValueError):
        TransformerResearchConfig(d_model=15, n_heads=2)


def test_past_sequences_look_backwards_only():
    b, ds = _bundle_and_dataset()
    builder = _InputBuilder(ds, b.prices, seq_len=10, target_horizons=(5, 20))
    obs = ds.included()[0]
    seq = builder.past_sequence(obs)
    assert len(seq) == 10
    dates, _channels = builder.daily[obs.ticker]
    used = [d for d in dates if d <= obs.decision_date][-10:]
    assert max(used) <= obs.decision_date  # never a bar after the decision date
    known = builder.known_future(obs)
    assert len(known) == 2 and known[-1][0] == 1.0  # horizon fraction of h_max


def test_preset_and_validation():
    gpu = TransformerResearchConfig.preset("gpu_4060")
    assert gpu.device == "cuda" and gpu.amp_bf16 and gpu.n_seeds == 5
    with pytest.raises(ValueError):
        TransformerResearchConfig.preset("nope")
    with pytest.raises(ValueError):
        TransformerResearchConfig(device="tpu")


@needs_torch
def test_multi_horizon_purge_refuses_overlapping_training():
    b, ds = _bundle_and_dataset()
    config = TransformerResearchConfig.preset("cpu_test", enabled=True, models=(MODEL_GRU,))
    (result,) = run_transformer_research(ds, horizon=5, config=config, prices=b.prices)
    # purge at h=60 removes all training rows: refuse rather than leak
    assert result.status == STATUS_INSUFFICIENT_DATA
    assert result.scored == []


def test_research_requires_prices():
    _b, ds = _bundle_and_dataset()
    if not torch_available():
        pytest.skip("torch not installed")
    results = run_transformer_research(
        ds, horizon=5, config=TransformerResearchConfig.preset("cpu_test", enabled=True)
    )
    assert {r.status for r in results} == {STATUS_INSUFFICIENT_DATA}


def test_pipeline_records_research_step_skipped_when_off(tmp_path):
    b, ds = _bundle_and_dataset()
    summary = run_pipeline(ds, b.prices, output_dir=tmp_path, run_id="r", fixed_timestamp=STAMP)
    step = next(s for s in summary["steps"] if s["name"] == "research_transformer_models")
    assert step["status"] == "skipped"
    assert "OFF" in step["skipped_reason"]
    assert not (tmp_path / "r" / "research").exists()


@needs_torch
@pytest.mark.parametrize("model_type", [MODEL_TFT, MODEL_GRU, MODEL_CS_TRANSFORMER])
def test_research_model_trains_oos_and_compares(model_type):
    b, ds = _bundle_and_dataset()
    reference = train_primary_model(ds, horizon=5, prices=b.prices)
    # the 60-row horizon would purge every training row on this 30-row-spaced
    # fixture (correctly -> insufficient_data), so train on 5/20
    config = TransformerResearchConfig.preset(
        "cpu_test", enabled=True, models=(model_type,), epochs=3, target_horizons=(5, 20)
    )
    (result,) = run_transformer_research(
        ds, horizon=5, config=config, prices=b.prices, reference_scored=reference.scored,
    )
    assert result.status == STATUS_TRAINED
    assert min(ds.decision_dates) not in {s.decision_date for s in result.scored}
    assert result.comparison["status"] == "ok"
    assert result.training["purge_horizon"] == 20
    if model_type == MODEL_TFT:
        assert "tft_static_variable_importance" in result.interpretation
        assert "tft_past_channel_importance" in result.interpretation
    (again,) = run_transformer_research(
        ds, horizon=5, config=config, prices=b.prices, reference_scored=reference.scored,
    )
    assert [s.score for s in result.scored] == [s.score for s in again.scored]


@needs_torch
def test_research_on_does_not_change_primary_outputs(tmp_path):
    b, ds = _bundle_and_dataset()
    run_pipeline(ds, b.prices, output_dir=tmp_path, run_id="off", fixed_timestamp=STAMP)
    run_pipeline(
        ds, b.prices, output_dir=tmp_path, run_id="on", fixed_timestamp=STAMP,
        config=PipelineConfig(research_transformers=True, research_preset="cpu_test"),
    )
    for rel in ("primary/predictions.csv", "ranking/ranking_metrics.json",
                "portfolio/portfolio_metrics.json"):
        off = (tmp_path / "off" / rel)
        on = (tmp_path / "on" / rel)
        if off.exists():
            assert off.read_bytes() == on.read_bytes(), rel
    assert (tmp_path / "on/research/transformers/tft/result.json").exists()


@pytest.mark.parametrize("horizon", [5, 20])
def test_pipeline_report_reuses_configured_linear_runs_and_primary_horizon(
    tmp_path, monkeypatch, horizon
):
    from jp_stock_analysis.modeling import report as report_module

    def unexpected_retrain(*args, **kwargs):
        raise AssertionError("report must reuse pipeline linear runs")

    monkeypatch.setattr(report_module, "_train_linear", unexpected_retrain)
    comparison_horizons = []
    cached_model = report_module._cached_model

    def capture_horizon(*args, **kwargs):
        comparison_horizons.append(kwargs["horizon"])
        return cached_model(*args, **kwargs)

    monkeypatch.setattr(report_module, "_cached_model", capture_horizon)
    bundle, dataset = _bundle_and_dataset()
    config = PipelineConfig(
        primary_model=MODEL_BASELINE,
        primary_horizon=horizon,
        report_optional_models=False,
        alpha=0.12,
        l1_ratio=0.8,
        portfolio_top_quantile=0.4,
        portfolio_bottom_quantile=0.3,
        portfolio_rank_weighted=True,
        transaction_cost_bps=17.0,
        neutralize_factors=("roe",),
    )
    run_pipeline(dataset, bundle.prices, output_dir=tmp_path, run_id="r", config=config)
    root = tmp_path / "r"

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    report = read("modeling_report.json")
    portfolio = read("portfolio/portfolio_metrics.json")
    neutral = read("neutralization/neutralized_metrics.json")
    assert comparison_horizons and set(comparison_horizons) == {horizon}
    assert report["model_comparison"]
    assert report["primary_model"]["horizon"] == horizon
    assert report["model_diversity"]["horizon"] == horizon
    assert report["neutralized_rank_metrics"] == neutral
    assert "roe" in neutral["exposure_columns"]
    summary = report["portfolio_long_short"][str(horizon)]
    assert summary["config"] == portfolio["config"]
    assert summary["mean_spread"] == portfolio["spread_series"]["mean_spread"]
    assert summary["config"]["weighting"] == "rank_weighted"
    for name in ("ridge", "elastic_net"):
        model = report["model_diversity"]["linear_models"][name]
        assert model["oos"] == read(f"linear/{name}/oos_validation.json")
        metadata_path = root / "linear" / name / "model_metadata.json"
        if metadata_path.exists():
            metadata = read(f"linear/{name}/model_metadata.json")
            assert model["coefficients"] == metadata["coefficients"]
            assert model["scaled_coefficients"] == metadata["scaled_coefficients"]
            if name == "elastic_net":
                assert metadata["alpha"] == 0.12
                assert metadata["l1_ratio"] == 0.8


def test_pipeline_report_does_not_retrain_disabled_linear_models(tmp_path, monkeypatch):
    from jp_stock_analysis.modeling import report as report_module

    def unexpected_retrain(*args, **kwargs):
        raise AssertionError("disabled models must not be retrained")

    monkeypatch.setattr(report_module, "_train_linear", unexpected_retrain)
    bundle, dataset = _bundle_and_dataset()
    run_pipeline(
        dataset, bundle.prices, output_dir=tmp_path, run_id="r",
        config=PipelineConfig(
            primary_model=MODEL_BASELINE, report_optional_models=False, linear_models=(),
        ),
    )
    report = json.loads((tmp_path / "r/modeling_report.json").read_text(encoding="utf-8"))
    for model in report["model_diversity"]["linear_models"].values():
        assert model["status"] == "disabled"
        assert model["oos"]["status"] == "disabled"
        assert model["oos"]["oos_scored_count"] == 0


@needs_trees
def test_tree_backend_cli_e2e_artifacts(tmp_path):
    """Exercise actual installed backends and validate persisted synthetic outputs."""
    import csv
    import math
    import os
    import subprocess
    import sys
    from pathlib import Path

    output = Path(os.environ.get("TREE_E2E_OUTPUT", str(tmp_path)))
    subprocess.run(
        [
            sys.executable, "-m", "jp_stock_analysis.cli", "run-modeling-pipeline",
            "--synthetic", "--primary-model", "tree_ensemble", "--tree-members",
            "lightgbm_regressor,catboost_regressor", "--tree-n-jobs", "1",
            "--run-id", "tree-e2e", "--fixed-timestamp", STAMP, "--output-dir", str(output),
        ],
        check=True, timeout=300,
    )
    root = output / "tree-e2e"

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    primary = read("primary/primary_model.json")
    assert primary["model_label"] == MODEL_TREE_ENSEMBLE
    assert primary["fallback_used"] is False
    assert primary["validation"] == "walk_forward_purged_oos"
    assert primary["scored_count"] > 0
    for name in ("lightgbm_regressor", "catboost_regressor"):
        member = primary["members"][name]
        assert member["status"] == "trained"
        assert member["validation"] == "walk_forward_purged_oos"
        assert member["scored_count"] > 0
        assert any(fold["status"] == "ok" for fold in member["oos_folds"])
        assert all(fold["train_end"] < fold["test_start"] for fold in member["oos_folds"])
    ridge = primary["members"]["ridge"]
    assert ridge["validation"] == "walk_forward_purged_oos"
    assert ridge["status"] == "ok" and ridge["oos_scored_count"] > 0
    assert any(fold["status"] == "ok" for fold in ridge["folds"])
    with (root / "primary/predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == primary["scored_count"]
    keys = [(row["ticker"], row["decision_date"]) for row in rows]
    assert len(set(keys)) == len(keys)
    assert all(math.isfinite(float(row["oos_score"])) for row in rows)
    assert min(build_synthetic_bundle().decision_dates).isoformat() not in {
        row["decision_date"] for row in rows
    }
    report = read("modeling_report.json")
    assert report["primary_model"]["model_label"] == primary["model_label"]
    assert report["primary_model"]["fallback_used"] is False
    assert read("ranking/ranking_metrics.json")["model_label"] == primary["model_label"]
    summary = read("pipeline_summary.json")
    assert summary["primary_model"] == primary["model_label"]
    steps = {step["name"]: step["status"] for step in summary["steps"]}
    assert all(steps[name] == "ok" for name in (
        "train_primary_model", "evaluate_ranking", "build_modeling_report",
    ))
