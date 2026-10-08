"""Общий расчёт метрик и календарных фолдов; обучение моделей пока не реализовано."""

import numpy as np
import pandas as pd


def development_origins(months, minimum_train_months=36, holdout_months=12,
                        max_horizon=12, step_months=1):
    """Origins такие, что даже target горизонта 12 не попадает в holdout.

    Месяцы должны образовывать непрерывную календарную шкалу. Пропуски отдельных
    территорий проверяются отдельно. Origin обозначает последний месяц контекста;
    timestamp реальной выдачи прогноза и available_at нужно проверить при fit.
    """
    periods = pd.PeriodIndex(pd.to_datetime(months), freq="M").unique().sort_values()
    if periods.empty:
        raise ValueError("Пустой календарь")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if not periods.equals(expected):
        raise ValueError("В календаре отсутствуют месяцы; сначала восстановите шкалу")
    if min(minimum_train_months, holdout_months, max_horizon, step_months) < 1:
        raise ValueError("Параметры должны быть положительными")
    holdout_start = periods[-holdout_months] if len(periods) >= holdout_months else periods[0]
    candidates = periods[minimum_train_months - 1::step_months]
    origins = candidates[candidates + max_horizon < holdout_start]
    return pd.DataFrame({"origin": origins.to_timestamp(),
                         "max_target": (origins + max_horizon).to_timestamp(),
                         "holdout_start": holdout_start.to_timestamp()})


def forecast_metrics(predictions: pd.DataFrame, series_column="territory_id"):
    """MAE по МО и наблюдениям, pooled R², coverage — отдельно для каждого h.

    Вход обязан содержать полную оценочную сетку, включая строки без прогноза.
    Неполное покрытие делает модель непригодной для честного leaderboard без
    заранее зафиксированного fallback. Сравнение моделей: на совпадающих ключах.
    """
    required = {series_column, "origin", "target_date", "horizon_months", "y_true", "y_pred"}
    if required - set(predictions):
        raise ValueError(f"Не хватает колонок: {sorted(required - set(predictions))}")
    df = predictions.copy()
    key = [series_column, "origin", "horizon_months"]
    if df.empty or df.duplicated(key).any():
        raise ValueError("Пустая таблица или повторные ключи прогноза")
    origins = pd.PeriodIndex(pd.to_datetime(df.origin), freq="M")
    targets = pd.PeriodIndex(pd.to_datetime(df.target_date), freq="M")
    horizons = pd.to_numeric(df.horizon_months, errors="raise")
    if ((horizons < 1) | (horizons % 1 != 0)).any():
        raise ValueError("Горизонт должен быть положительным целым числом месяцев")
    expected = pd.PeriodIndex([o + int(h) for o, h in zip(origins, horizons)], freq="M")
    if not targets.equals(expected):
        raise ValueError("target_date не соответствует календарному горизонту")
    df["y_true"] = pd.to_numeric(df.y_true, errors="raise")
    df["y_pred"] = pd.to_numeric(df.y_pred, errors="raise")
    if not np.isfinite(df.y_true.to_numpy(dtype=float)).all():
        raise ValueError("Оценочная сетка должна содержать конечный y_true")
    valid = np.isfinite(df.y_pred.to_numpy(dtype=float))
    df["valid_prediction"] = valid
    df["absolute_error"] = (df.y_true - df.y_pred).abs().where(valid)
    rows = []
    for horizon, group in df.groupby("horizon_months"):
        observed = group[group.valid_prediction]
        if observed.empty:
            macro = micro = r2 = np.nan
        else:
            macro = observed.groupby(series_column).absolute_error.mean().mean()
            micro = observed.absolute_error.mean()
            denominator = ((observed.y_true - observed.y_true.mean()) ** 2).sum()
            r2 = 1 - ((observed.y_true - observed.y_pred) ** 2).sum() / denominator if denominator > 0 else np.nan
        rows.append({"horizon_months": int(horizon), "n_series": group[series_column].nunique(),
                     "n_predictions": len(observed), "n_expected": len(group),
                     "macro_mae": macro, "micro_mae": micro, "pooled_r2": r2,
                     "coverage": group.valid_prediction.mean()})
    return pd.DataFrame(rows)
