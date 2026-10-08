from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.news_impact import news_inputs, amplify_financial_input


def panel():
    return SimpleNamespace(n_series=2, n_months=4,
                           months=pd.date_range('2024-01-01', periods=4, freq='MS'),
                           meta=pd.DataFrame({'territory_id': [10, 20]}))


def records(date='2024-02-06T08:42:00+03:00'):
    return pd.DataFrame([{'duplicate_key': 'a', 'historical_available_at': date,
                          'relevant': True, 'event_region_confirmed_in_title': True,
                          'territory_ids': 10.0, 'topic': 'flood_or_evacuation'}])


def test_local_news_with_numeric_csv_id_does_not_spread_to_other_mo():
    features, _, signal, _ = news_inputs(panel(), records())
    assert signal[0, 1] == 1
    assert not signal[1].any()
    assert features[1, 0, 4] > 0
    assert features[1, 1, 4] == 0


def test_future_news_cannot_change_earlier_features_and_placebo_only_delays():
    original, _, signal, _ = news_inputs(panel(), records())
    later = records('2024-04-06T08:42:00+03:00').assign(duplicate_key='b')
    combined, _, _, _ = news_inputs(panel(), pd.concat([records(), later]))
    np.testing.assert_array_equal(original[:3], combined[:3])
    delayed, _, delayed_signal, _ = news_inputs(panel(), records(), delay_months=3)
    assert not delayed_signal.any()
    assert signal[0, 1]


def test_detector_news_input_preserves_sign_and_has_no_signal_without_financial_residual():
    z = np.array([[0., -2., np.nan]])
    result = amplify_financial_input(z, np.ones_like(z), .5)
    assert result[0, 0] == 0
    assert result[0, 1] == -3
    assert np.isnan(result[0, 2])
