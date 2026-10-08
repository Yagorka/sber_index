"""Последовательные сравнения: метки подбора и калибровки доступны на origin.

Алгоритмы заданы после просмотра исходного теста. As-of здесь относится к
численным входам, а не к независимости дизайна эксперимента или vintages.
"""
import numpy as np
from src.mun_ensemble import simplex_lad_weights


def historical_pairs(origin, horizon, min_origin=5):
    return [(a, a + horizon) for a in range(min_origin, origin - horizon + 1)]


def select_past_weights(values, predictions, origin, horizon, calibration_months=2,
                        minimum_months=2, seed=42, max_rows=3000):
    names = list(predictions)
    pairs = [(a, t) for a, t in historical_pairs(origin, horizon)
             if t <= origin - calibration_months]
    if len(pairs) < minimum_months:
        return dict.fromkeys(names, 1 / len(names)), [], "fixed_equal_no_mature_selection_history"
    y = np.concatenate([values[:, t] for _, t in pairs])
    P = np.column_stack([np.concatenate([predictions[n][a, horizon] for a, _ in pairs]) for n in names])
    if not np.isfinite(P).all():
        raise ValueError("Все кандидаты должны покрывать историческую сетку")
    w = simplex_lad_weights(y, P, seed=seed, max_rows=max_rows)
    return dict(zip(names, map(float, w))), pairs, "past_only_lad"


def rolling_blend(values, predictions, horizons=(1, 3, 6, 12), calibration_months=2,
                  minimum_months=2, seed=42, max_rows=3000):
    out = np.full_like(next(iter(predictions.values())), np.nan, dtype=float)
    audit = []
    for o in range(5, values.shape[1]):
        for h in horizons:
            weights, pairs, mode = select_past_weights(values, predictions, o, h,
                                                       calibration_months, minimum_months, seed, max_rows)
            out[o, h] = sum(w * predictions[n][o, h] for n, w in weights.items())
            audit.append({"origin": o, "h": h, "mode": mode,
                          "selection_last_target": max((t for _, t in pairs), default=-1),
                          "selection_months": len(pairs), **{f"weight_{n}": w for n, w in weights.items()}})
    return out, audit


def past_calibration(values, prequential_prediction, categories, origin, horizon,
                     calibration_months=2, minimum_months=2, coverage=0.9):
    """Ошибки действительно выданных ранее прогнозов; цели не использованы для текущих весов.

Не предполагает обменность территорий или временных ошибок; покрытие эмпирическое.
"""
    pairs = [(a, t) for a, t in historical_pairs(origin, horizon)
             if origin - calibration_months < t <= origin]
    q = np.full(values.shape[0], np.nan)
    if len(pairs) < minimum_months:
        return q, pairs
    p = np.stack([prequential_prediction[a, horizon] for a, _ in pairs])
    y = np.stack([values[:, t] for _, t in pairs])
    e = np.abs(y - p) / np.maximum(p, 1e-9)
    for c in np.unique(categories):
        z = e[:, categories == c].ravel()
        z = z[np.isfinite(z)]
        if len(z):
            level = min(1., np.ceil((len(z) + 1) * coverage) / len(z))
            q[categories == c] = np.quantile(z, level, method="higher")
    return q, pairs


def shock_adjustment(prediction, residuals, alarms, shrink=0.5, cap=0.3, decay_months=3):
    """После наблюдения тревоги используем только последний доступный остаток.

Коррекция будущего уровня; ни текущий факт, ни будущие тревоги не подменяют прогноз.
"""
    out = prediction.copy()
    last = np.full(residuals.shape[0], -1)
    for o in range(5, residuals.shape[1]):
        last[alarms[:, o]] = o
        active = (last >= 0) & (o - last < decay_months)
        correction = np.zeros(len(last))
        ids = np.where(active)[0]
        correction[ids] = (shrink * np.clip(residuals[ids, last[ids]], -cap, cap)
                           * (1 - (o - last[ids]) / decay_months))
        for h in range(1, prediction.shape[1]):
            out[o, h] = prediction[o, h] * np.exp(correction)
    return out
