"""Point-in-time availability of disclosure text in modeling datasets."""

from datetime import date

import pytest
from pydantic import ValidationError

from jp_stock_analysis.modeling.dataset import build_modeling_dataset
from jp_stock_analysis.modeling.fixtures import build_synthetic_bundle
from jp_stock_analysis.schemas import DisclosureDocument


def _dataset(document):
    bundle = build_synthetic_bundle()
    ticker = sorted(bundle.narratives)[0]
    narratives = dict(bundle.narratives)
    narratives[ticker] = document
    dataset = build_modeling_dataset(
        bundle.fundamentals, bundle.prices, bundle.metadata, narratives,
        decision_dates=bundle.decision_dates, horizons=bundle.horizons,
        bundle_disclosure_date=bundle.bundle_disclosure_date, is_synthetic=True,
    )
    return ticker, dataset


def _doc(**kwargs):
    return DisclosureDocument(ticker="1301", text="為替リスクと減損", **kwargs)


@pytest.mark.parametrize(
    ("available", "expected"),
    [(date(2025, 2, 11), 1.0), (date(2025, 2, 12), 0.0),
     (date(2025, 2, 13), 0.0), (None, 0.0)],
)
def test_pit_date_boundary(available, expected):
    ticker, ds = _dataset(_doc(available_date=available))
    obs = next(o for o in ds.observations if o.ticker == ticker)
    assert obs.features["narrative_available"] == expected
    assert (obs.features["risk_keyword_count"] is not None) == bool(expected)
    assert obs.features["sentiment_placeholder"] == (0.0 if expected else None)


def test_unknown_metadata_does_not_authorize_narrative():
    ticker, ds = _dataset(_doc(source_metadata={"target_date": "2099-01-01"}))
    assert all(
        o.features["narrative_available"] == 0.0
        for o in ds.observations if o.ticker == ticker
    )


def test_empty_narrative_stays_missing():
    doc = DisclosureDocument(ticker="1301", text="  ", available_date=date(2020, 1, 1))
    ticker, ds = _dataset(doc)
    obs = next(o for o in ds.observations if o.ticker == ticker)
    assert (obs.features["narrative_available"], obs.features["risk_keyword_count"],
            obs.features["sentiment_placeholder"]) == (0.0, None, None)


def test_invalid_date_is_rejected():
    with pytest.raises(ValidationError):
        _doc(available_date="not-a-date")


def test_state_changes_by_date_without_changing_other_features_or_labels():
    ticker, with_text = _dataset(_doc(available_date=date(2025, 3, 26)))
    _, without_text = _dataset(None)
    narrative_fields = {"narrative_available", "risk_keyword_count", "sentiment_placeholder"}
    before = [o for o in without_text.observations if o.ticker == ticker]
    after = [o for o in with_text.observations if o.ticker == ticker]
    assert [o.features["narrative_available"] for o in after] == [0.0, 0.0, 1.0]
    for a, b in zip(before, after, strict=True):
        assert {k: v for k, v in a.features.items() if k not in narrative_fields} == {
            k: v for k, v in b.features.items() if k not in narrative_fields
        }
        assert a.labels == b.labels
        assert a.included == b.included
    assert with_text.feature_names == without_text.feature_names
