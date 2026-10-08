"""Проверки муниципального блока на синтетике: временные границы, утечки, метрики."""

import unittest

import numpy as np
import pandas as pd

from src.mun_data import CATEGORIES, NATIONAL_COLUMNS, MunicipalPanel, NationalPriors
from src.mun_eval import cluster_bootstrap_diff, make_grid, score_model
from src.mun_models import Context, feature_matrix, simple_baselines, training_set
from src.mun_shocks import category_scales, residual_matrix, standardize, run_detector, match_alarms


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    nat_idx = pd.date_range("2018-12-01", periods=84, freq="MS")
    t = np.arange(84)
    nat = pd.DataFrame({c: 100 * np.exp(0.01 * t + 0.05 * np.sin(2 * np.pi * t / 12 + k))
                        for k, c in enumerate(NATIONAL_COLUMNS)}, index=nat_idx)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows, values = [], []
    for terr in range(1, 5):
        for c in CATEGORIES:
            rows.append({"territory_id": terr, "category": c, "series_id": f"{terr}|{c}",
                         "region_code": terr % 2, "region_name": f"r{terr % 2}", "mo_type": "x", "mo_name": f"mo{terr}",
                         "lat": 50. + terr, "lon": 30. + terr, "market_access": 300. + terr, "mo_type_code": 1})
            k = rng.uniform(900, 3000)
            values.append(k * np.exp(0.012 * np.arange(24) + 0.04 * np.sin(2 * np.pi * np.arange(24) / 12)
                                     + rng.normal(0, 0.02, 24)))
    panel = MunicipalPanel(np.array(values), pd.DataFrame(rows), months)
    return panel, NationalPriors(nat)


class MunicipalLeakageTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.nat = make_world()

    def test_grid_targets_stay_in_stage_and_history_minimum(self):
        grid = make_grid()
        self.assertTrue((grid.origin >= 5).all())
        self.assertTrue(((grid.stage == "validation") == grid.target.between(12, 17)).all())
        self.assertTrue(((grid.stage == "test") == grid.target.between(18, 23)).all())
        self.assertTrue((grid.target - grid.origin == grid.h).all())

    def test_future_values_do_not_change_features_or_baselines(self):
        ctx = Context(self.panel, self.nat)
        origin = 11
        X0 = [feature_matrix(ctx, origin, h)[0] for h in (1, 6, 12)]
        base0 = simple_baselines(ctx, origin)
        altered = MunicipalPanel(self.panel.values.copy(), self.panel.meta, self.panel.months)
        altered.values[:, origin + 1:] *= 50
        altered.log = np.log(altered.values)
        ctx2 = Context(altered, self.nat)
        for h, x in zip((1, 6, 12), X0):
            np.testing.assert_array_equal(x, feature_matrix(ctx2, origin, h)[0])
        base1 = simple_baselines(ctx2, origin)
        for name in base0:
            np.testing.assert_allclose(base0[name], base1[name])

    def test_national_future_does_not_change_priors(self):
        ctx = Context(self.panel, self.nat)
        a, h = 11, 12
        path = ctx.nat_path(a, h)
        nat2 = self.nat.panel.copy()
        # данные национальных рядов после месяца якоря искажены
        anchor_date = self.panel.months[a]
        nat2.loc[nat2.index > anchor_date] *= 7
        ctx2 = Context(self.panel, NationalPriors(nat2))
        np.testing.assert_allclose(path, ctx2.nat_path(a, h))
        np.testing.assert_allclose(ctx.nat_growth(a), ctx2.nat_growth(a))

    def test_training_labels_not_after_origin(self):
        ctx = Context(self.panel, self.nat)
        origin = 14
        _, y, w, meta = training_set(ctx, origin)
        self.assertTrue(all(a + h <= origin for a, h in meta))
        self.assertTrue(all(a >= 5 for a, h in meta))
        altered = MunicipalPanel(self.panel.values.copy(), self.panel.meta, self.panel.months)
        altered.values[:, origin + 1:] *= 99
        altered.log = np.log(altered.values)
        _, y2, w2, _ = training_set(Context(altered, self.nat), origin)
        np.testing.assert_allclose(y, y2)
        np.testing.assert_allclose(w, w2)

    def test_standardization_uses_only_past(self):
        ctx = Context(self.panel, self.nat)
        r = residual_matrix(ctx)
        cats = self.panel.category_code
        ref = category_scales(r, cats)
        z = standardize(r, cats, ref)
        t = 15
        r2 = r.copy()
        r2[:, t + 1:] += 5.
        ref2 = category_scales(r2, cats)
        z2 = standardize(r2, cats, ref2)
        np.testing.assert_allclose(z[:, :t + 1], z2[:, :t + 1], equal_nan=True)


class MetricsAndDetectorTests(unittest.TestCase):
    def test_perfect_forecast_scores(self):
        panel, nat = make_world()
        grid = make_grid()
        pred = np.full((24, 13, panel.n_series), np.nan)
        for o in range(5, 23):
            for h in range(1, 13):
                if o + h <= 23:
                    pred[o, h] = panel.values[:, o + h]
        res = score_model(panel, pred, grid, "test", "oracle")
        self.assertTrue((res.mae < 1e-9).all())
        self.assertTrue((res.r2 > 0.999999).all())

    def test_bootstrap_ci_contains_zero_for_identical_models(self):
        a = np.abs(np.random.default_rng(1).normal(size=400))
        clusters = np.repeat(np.arange(100), 4)
        d, lo, hi = cluster_bootstrap_diff(a, a.copy(), clusters, 200)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))

    def test_detectors_detect_level_shift_and_one_to_one_matching(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=60)
        z[40:] += 3.0
        for method, thr in [("CUSUM", 4), ("PageHinkley", 5), ("BOCPD", 0.5)]:
            alarms = np.where(run_detector(z, method, thr))[0]
            self.assertTrue(any(40 <= a <= 46 for a in alarms), method)
        self.assertEqual(match_alarms(40, [38, 42, 43], 3), 42)
        self.assertIsNone(match_alarms(40, [38, 50], 3))


if __name__ == "__main__":
    unittest.main()
