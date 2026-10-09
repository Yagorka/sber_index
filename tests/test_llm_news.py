from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.llm_news import aggregate_features, feature_names, select_sample, validate_rows
from scripts.evaluate_llm_news import select_validation, fit_corrections
from src.mun_eval import make_grid


def panel():
    return SimpleNamespace(n_months=4, n_series=3, category_code=np.array([0, 1, 5]),
                           months=pd.date_range('2024-01-01', periods=4, freq='MS'),
                           meta=pd.DataFrame({'territory_id': [10, 10, 20]}))


def record(**changes):
    row = {'duplicate_key': 'a', 'historical_available_at': '2024-02-05T10:00:00+03:00',
           'sampling_weight': 1., 'economic_relevance': 1, 'sentiment': -1,
           'price_direction': 1, 'income_direction': 9, 'business_direction': 9,
           'category_mask': 2, 'shock': 0, 'event_region_confirmed_in_title': True,
           'territory_ids': '10', 'geo_status': 'matched_title_alias'}
    row.update(changes)
    return row


def test_strict_labels_reject_missing_fields_and_invalid_unknown_codes():
    assert validate_rows({'rows': [[0, 0, 1, 1, 9, 9, 2, 0]]}, 1)[0]['income_direction'] == 9
    with pytest.raises(ValueError):
        validate_rows({'rows': [[0, 0, 1, 1, 9, 9, 2]]}, 1)
    with pytest.raises(ValueError):
        validate_rows({'rows': [[0, 0, 1, 7, 9, 9, 2, 0]]}, 1)


def test_local_category_signal_does_not_spread_and_unknown_is_not_neutral():
    features, _ = aggregate_features(panel(), pd.DataFrame([record()]))
    names = feature_names()
    assert features[1, 1, names.index('price_up_share_1m')] == 1
    assert features[1, 1, names.index('income_known_share_1m')] == 0
    assert features[1, 2, names.index('price_up_share_1m')] == 0
    assert features[1, 2, names.index('economic_share_1m')] == 0
    neutral, _ = aggregate_features(panel(), pd.DataFrame([record(income_direction=0)]))
    assert neutral[1, 1, names.index('income_known_share_1m')] == 1


def test_future_news_cannot_change_past_features_and_delay_only_moves_forward():
    one = pd.DataFrame([record()])
    original, _ = aggregate_features(panel(), one)
    future = record(duplicate_key='b', historical_available_at='2024-04-05T10:00:00+03:00', sentiment=1)
    combined, _ = aggregate_features(panel(), pd.concat([one, pd.DataFrame([future])]))
    np.testing.assert_array_equal(original[:3], combined[:3])
    delayed, _ = aggregate_features(panel(), one, delay_months=3)
    assert not delayed.any()


def test_unresolved_location_and_housing_without_category_do_not_become_spending_signal():
    names = feature_names()
    for row in [record(territory_ids='', geo_status='ambiguous_or_unresolved'), record(category_mask=0)]:
        features, _ = aggregate_features(panel(), pd.DataFrame([row]))
        assert not features[:, :, names.index('log_aligned_economic_articles_1m')].any()


def test_sample_strata_weight_back_to_original_month_and_do_not_use_spending():
    rows = pd.DataFrame([{'duplicate_key': str(i), 'title': f'News {i}', 'relevant': i < 6,
                          'historical_available_at': '2024-01-05T10:00:00+03:00'} for i in range(10)])
    cfg = {'seed': 42, 'sample_per_month_relevant': 2, 'sample_per_month_other': 2}
    sample = select_sample(rows, cfg)
    assert len(sample) == 4 and sample.sampling_weight.sum() == 10
    reversed_sample = select_sample(rows.iloc[::-1], cfg)
    assert sample.title_sha256.tolist() == reversed_sample.title_sha256.tolist()


def test_selection_ignores_test_targets_and_allows_zero_correction():
    months = np.arange(24, dtype=float)
    p = SimpleNamespace(values=np.vstack([months**2*.001+100., months**2*.002+101.]))
    base = np.full((24, 13, 2), 100.)
    candidates = {('llm', 100, 0.): base, ('llm', 100, .5): base*1.1}
    selected, choice, _ = select_validation(p, candidates, make_grid())
    assert (choice.shrink == 0).all()
    p.values[:, 18:] *= 100
    _, changed_choice, _ = select_validation(p, candidates, make_grid())
    pd.testing.assert_frame_equal(choice, changed_choice)
    np.testing.assert_array_equal(selected['llm'], base)


def test_future_financial_targets_do_not_change_already_issued_corrections():
    p = SimpleNamespace(values=np.tile(np.arange(24, dtype=float)+100., (2, 1)),
                        months=pd.date_range('2023-01-01', periods=24, freq='MS'),
                        category_code=np.array([0, 1]), n_series=2)
    base = np.full((24, 13, 2), 100.)
    cfg = {'ridge_alphas': [100], 'correction_shrinks': [.1], 'minimum_training_pairs': 2,
           'maximum_log_correction': .15}
    original, audits = fit_corrections(p, base, {'llm': np.zeros((24, 2, 2))}, cfg)
    p.values[:, 19:] *= 10
    changed, _ = fit_corrections(p, base, {'llm': np.zeros((24, 2, 2))}, cfg)
    np.testing.assert_array_equal(original['llm', 100, .1][:19], changed['llm', 100, .1][:19])
    assert all(row['last_training_target'] <= row['origin'] for row in audits)


def test_zero_baseline_is_not_used_in_log_training_and_remains_zero():
    p = SimpleNamespace(values=np.tile(np.arange(24, dtype=float)+100., (2, 1)),
                        months=pd.date_range('2023-01-01', periods=24, freq='MS'),
                        category_code=np.array([0, 1]), n_series=2)
    base = np.full((24, 13, 2), 100.)
    base[:, :, 0] = 0
    cfg = {'ridge_alphas': [100], 'correction_shrinks': [.1], 'minimum_training_pairs': 2,
           'maximum_log_correction': .15}
    predictions, audits = fit_corrections(p, base, {'llm': None}, cfg)
    assert np.isfinite(predictions['llm', 100, .1]).all()
    assert not predictions['llm', 100, .1][:, :, 0].any()
    assert any(row['excluded_nonpositive_training_forecasts'] > 0 for row in audits)
