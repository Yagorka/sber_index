"""Сетка origin'ов, метрики и парный bootstrap для муниципальной панели.

Индексы месяцев 0..23 = 2023-01..2024-12. Origin — последний месяц контекста,
целевой месяц = origin + h. Протокол (правила организатора мне неизвестны, это
явное допущение «конкурсного» протокола):
  validation: цели 2024-01..2024-06 (индексы 12..17), для выбора моделей и весов;
  test:       цели 2024-07..2024-12 (индексы 18..23), открывается после заморозки выбора.
Минимальная история в origin — 6 месяцев (индекс origin >= 5).
"""

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)
MIN_ORIGIN = 5
STAGES = {"validation": (12, 17), "test": (18, 23)}


def make_grid(n_months=24, horizons=HORIZONS, min_origin=MIN_ORIGIN):
    rows = []
    for stage, (lo, hi) in STAGES.items():
        for target in range(lo, hi + 1):
            for h in horizons:
                origin = target - h
                if origin >= min_origin:
                    rows.append({"stage": stage, "origin": origin, "h": h, "target": target})
    grid = pd.DataFrame(rows)
    assert (grid.target <= n_months - 1).all() and (grid.origin >= min_origin).all()
    return grid


def pair_errors(panel, pred, grid, stage):
    """Возвращает (y_true, y_pred, origin_value, h) в виде массивов формы (n_pairs, S)."""
    g = grid[grid.stage == stage]
    y = np.stack([panel.values[:, t] for t in g.target])
    p = np.stack([pred[o, h] for o, h in zip(g.origin, g.h)])
    base = np.stack([panel.values[:, o] for o in g.origin])
    return g.reset_index(drop=True), y, p, base


def score_model(panel, pred, grid, stage, name):
    g, y, p, base = pair_errors(panel, pred, grid, stage)
    rows = []
    for h in sorted(g.h.unique()):
        m = (g.h == h).to_numpy()
        yt, yp, yb = y[m], p[m], base[m]
        ok = np.isfinite(yp)
        err = np.abs(yt - yp)
        sse = np.nansum((yt - yp) ** 2)
        den = np.sum((yt - yt.mean()) ** 2)
        dch = yt - yb
        den_c = np.sum((dch - dch.mean()) ** 2)
        sse_c = np.nansum((dch - (yp - yb)) ** 2)
        rows.append({"model": name, "stage": stage, "h": int(h), "n_target_months": int(m.sum()),
                     "n_series": y.shape[1], "mae": float(np.nanmean(err)),
                     "r2": float(1 - sse / den), "r2_change": float(1 - sse_c / den_c),
                     "wape": float(np.nansum(err) / np.sum(yt)), "coverage": float(ok.mean())})
    return pd.DataFrame(rows)


def per_series_mae(panel, pred, grid, stage, h):
    g, y, p, _ = pair_errors(panel, pred, grid, stage)
    m = (g.h == h).to_numpy()
    return np.abs(y[m] - p[m]).mean(0)


def cluster_bootstrap_diff(a, b, clusters, replicates=1000, seed=42):
    """Парный bootstrap разности MAE (a - b) по кластерам-МО; a, b — MAE на ряд."""
    codes, inverse = np.unique(clusters, return_inverse=True)
    n = len(codes)
    sums_d = np.bincount(inverse, weights=a - b, minlength=n)
    counts = np.bincount(inverse, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        pick = rng.integers(0, n, n)
        draws[i] = sums_d[pick].sum() / counts[pick].sum()
    full = float((a - b).mean())
    return full, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def month_bootstrap_diff(panel, pred_a, pred_b, grid, stage, h, replicates=1000, seed=42):
    """Bootstrap по целевым месяцам (общий шок месяца не усредняется по МО)."""
    g, y, pa, _ = pair_errors(panel, pred_a, grid, stage)
    _, _, pb, _ = pair_errors(panel, pred_b, grid, stage)
    m = (g.h == h).to_numpy()
    d = (np.abs(y[m] - pa[m]) - np.abs(y[m] - pb[m])).mean(1)
    rng = np.random.default_rng(seed)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(replicates)]
    return float(d.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), len(d)
