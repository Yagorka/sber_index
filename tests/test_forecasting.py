"""Проверки временных границ: выявляют утечку, а не повторяют формулы модели."""

import unittest
import numpy as np
import pandas as pd
from src.forecasting import make_split, direct_features, direct_training, fit_predict


class ForecastingValidationTests(unittest.TestCase):
    def setUp(self):
        n = 93
        self.panel = pd.DataFrame({"a": 100 + np.arange(n) + 10 * np.sin(np.arange(n) * np.pi / 6),
                                   "b": 50 + np.arange(n) * .5},
                                  index=pd.date_range("2018-12-01", periods=n, freq="MS"))

    def test_all_development_targets_are_before_holdout(self):
        validation = {"horizons_months": [1, 3, 6, 12], "minimum_train_months": 36,
                      "holdout_months": 12, "origin_step_months": 1}
        split, dev, test, start = make_split(self.panel, validation)
        self.assertEqual(start, 81)
        self.assertEqual(len(set(o for o, h in dev)), 34)
        self.assertTrue(all(o + h < start for o, h in dev))
        self.assertTrue(all(start <= o + h < len(self.panel) for o, h in test))
        self.assertTrue((split.train_months == split.origin_index + 1).all())

    def test_future_values_do_not_change_features_or_fit(self):
        altered = self.panel.copy()
        altered.iloc[51:] *= 1000
        for h in [1, 3, 6, 12]:
            _, _, labels = direct_training(self.panel, 50, h)
            self.assertLessEqual(labels.max(), 50)
            for j, series in enumerate(self.panel.columns):
                x, ref = direct_features(self.panel[series].to_numpy(), 50, h, j, 2)
                other, ref_other = direct_features(altered[series].to_numpy(), 50, h, j, 2)
                np.testing.assert_array_equal(x, other)
                self.assertEqual(ref, ref_other)
        variant = {"id": "ridge", "family": "RidgeDirect", "params": {"alpha": 1}}
        expected, _, _ = fit_predict(self.panel, 50, [1, 3, 6, 12], variant)
        changed, _, _ = fit_predict(altered, 50, [1, 3, 6, 12], variant)
        self.assertEqual(expected, changed)


if __name__ == "__main__":
    unittest.main()
