"""Национальный backtest. Все direct-метки и признаки ограничены origin.

Текущая версия истории не содержит available_at: это latest-vintage backtest
с явным допущением доступности значения контекстного месяца, не real-time тест.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.consumer import read_export
from src.evaluation import forecast_metrics


def load_panel(path):
    raw, _ = read_export(path)
    if set(raw.ref_area) != {"Россия"} or set(raw.unit_measure) != {"млрд. руб."}:
        raise ValueError("Ожидаются месячные расходы России в млрд руб.")
    if raw.duplicated(["period", "type"]).any():
        raise ValueError("Повторные ключи источника")
    panel = raw.pivot(index="period", columns="type", values="value").sort_index()
    panel.index = pd.to_datetime(panel.index)
    if not panel.index.equals(pd.date_range(panel.index.min(), panel.index.max(), freq="MS")):
        raise ValueError("Календарь должен быть непрерывным")
    if not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any():
        raise ValueError("Нужны конечные положительные расходы")
    return panel


def make_split(panel, validation):
    horizons = validation["horizons_months"]
    minimum = validation["minimum_train_months"]
    holdout_start = len(panel) - validation["holdout_months"]
    dev_origins = list(range(minimum - 1, holdout_start - max(horizons),
                             validation["origin_step_months"]))
    if not dev_origins:
        raise ValueError("Истории недостаточно для train/dev/holdout")
    dev_pairs = [(origin, h) for origin in dev_origins for h in horizons]
    test_pairs = [(target - h, h) for target in range(holdout_start, len(panel)) for h in horizons]
    if min(o for o, _ in test_pairs) < minimum - 1:
        raise ValueError("Недостаточно train для тестовых горизонтов")
    records = []
    for stage, pairs in [("development", dev_pairs), ("holdout", test_pairs)]:
        for origin, h in pairs:
            records.append({"stage": stage, "origin_index": origin, "origin": panel.index[origin],
                            "target_index": origin + h, "target_date": panel.index[origin + h],
                            "horizon_months": h, "train_months": origin + 1})
    return pd.DataFrame(records), dev_pairs, test_pairs, holdout_start


def seasonal_reference(values, origin, h):
    # h <= 12: значение того же месяца прошлого года уже известно.
    if not 1 <= h <= 12:
        raise ValueError("Горизонт должен быть в 1..12")
    return float(values[origin + h - 12])


def direct_features(values, anchor, h, series_index, n_series):
    if anchor < 23:
        raise ValueError("Для признаков нужно 24 месяца контекста")
    past = np.asarray(values[:anchor + 1], dtype=float)
    scale = past[-12:].mean()
    reference = seasonal_reference(past, anchor, h)
    features = [past[-1 - lag] / scale for lag in [0, 1, 2, 3, 6, 11, 12, 23]]
    for window in [3, 6, 12]:
        features += [past[-window:].mean() / scale, past[-window:].std() / scale]
    features += [scale / past[-24:-12].mean(), reference / scale, anchor / 12]
    month_phase = 2 * np.pi * ((anchor + h) % 12) / 12
    features += [np.sin(month_phase), np.cos(month_phase)]
    features += [float(j == series_index) for j in range(n_series)]
    return np.asarray(features), reference


def direct_training(panel, origin, h):
    x, y, labels = [], [], []
    for series_index, series in enumerate(panel.columns):
        values = panel[series].to_numpy()
        for anchor in range(23, origin - h + 1):
            features, reference = direct_features(values, anchor, h, series_index, len(panel.columns))
            x.append(features)
            # Цель относительная к известной сезонной базе; рост можно экстраполировать.
            y.append(values[anchor + h] / reference - 1)
            labels.append(anchor + h)
    if not labels or max(labels) > origin:
        raise ValueError("Нарушена граница доступности direct-меток")
    return np.asarray(x), np.asarray(y), np.asarray(labels)


def fourier_matrix(times, center, harmonics):
    times = np.asarray(times, dtype=float)
    cols = [(times - center) / 12]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * times / 12), np.cos(2 * np.pi * k * times / 12)]
    return np.column_stack(cols)


def fit_predict(panel, origin, horizons, variant, seed=42):
    """Возвращает (h, series, value), fitted-объекты и audit доступности меток."""
    family, params = variant["family"], variant["params"]
    records, fitted, audit = [], {}, []
    if family in ["RidgeDirect", "RandomForestDirect", "CatBoostDirect"]:
        for h in horizons:
            x, y, labels = direct_training(panel, origin, h)
            if family == "RidgeDirect":
                model = make_pipeline(StandardScaler(), Ridge(**params))
            elif family == "RandomForestDirect":
                model = RandomForestRegressor(**params, random_state=seed, n_jobs=1)
            else:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(**params, loss_function="MAE", random_seed=seed,
                                          thread_count=1, verbose=False, allow_writing_files=False)
            model.fit(x, y)
            fitted[h] = model
            audit.append({"origin_index": origin, "horizon_months": h,
                          "max_training_target_index": int(labels.max()), "training_examples": len(y)})
            for j, series in enumerate(panel.columns):
                features, reference = direct_features(panel[series].to_numpy(), origin, h, j, len(panel.columns))
                value = reference * (1 + float(model.predict(features.reshape(1, -1))[0]))
                records.append((h, series, max(0., value)))
        return records, fitted, audit
    for series in panel.columns:
        past = panel[series].iloc[:origin + 1].to_numpy()
        if family == "FourierRidge":
            start = max(0, len(past) - params["window"])
            times = np.arange(start, len(past))
            center = times.mean()
            model = Ridge(alpha=params["alpha"])
            model.fit(fourier_matrix(times, center, params["harmonics"]), np.log(past[start:]))
            fitted[series] = {"model": model, "center": center, "harmonics": params["harmonics"]}
        for h in horizons:
            reference = seasonal_reference(past, origin, h)
            if family == "LastValue":
                value = past[-1]
            elif family == "SeasonalNaive":
                value = reference
            elif family == "SeasonalGrowth":
                value = reference * (past[-12:].mean() / past[-24:-12].mean())
            elif family == "FourierRidge":
                value = np.exp(model.predict(fourier_matrix([origin + h], center, params["harmonics"]))[0])
            else:
                raise ValueError(f"Неизвестная модель: {family}")
            records.append((h, series, max(0., float(value))))
    return records, fitted, audit


def score_predictions(predictions):
    metrics, per_series = [], []
    for (stage, variant, family), group in predictions.groupby(["stage", "variant", "model"]):
        current = forecast_metrics(group, series_column="series_id")
        current["stage"], current["variant"], current["model"] = stage, variant, family
        metrics.append(current)
        for (series, h), part in group.groupby(["series_id", "horizon_months"]):
            denom = ((part.y_true - part.y_true.mean()) ** 2).sum()
            per_series.append({"stage": stage, "variant": variant, "model": family, "series_id": series,
                               "horizon_months": h, "mae": (part.y_true - part.y_pred).abs().mean(),
                               "r2": 1 - ((part.y_true - part.y_pred) ** 2).sum() / denom if denom else np.nan,
                               "n_predictions": len(part)})
    return pd.concat(metrics, ignore_index=True), pd.DataFrame(per_series)


def inverse_error_ensemble(predictions, chosen_variants, weights, name="DevWeightedEnsemble"):
    selected = predictions[predictions.variant.isin(chosen_variants)].copy()
    selected["weight"] = [weights[str(int(h))][v] for h, v in zip(selected.horizon_months, selected.variant)]
    selected["weighted_prediction"] = selected.y_pred * selected.weight
    keys = ["stage", "origin", "target_date", "horizon_months", "series_id"]
    aggregated = selected.groupby(keys).agg(y_true=("y_true", "first"), y_pred=("weighted_prediction", "sum"),
                                           n_members=("variant", "nunique"), total_weight=("weight", "sum")).reset_index()
    if (aggregated.n_members != len(chosen_variants)).any() or not np.allclose(aggregated.total_weight, 1):
        raise ValueError("Неполный состав ансамбля")
    aggregated["variant"] = "ensemble"
    aggregated["model"] = name
    aggregated["fallback"] = False
    return aggregated.drop(columns=["n_members", "total_weight"])


def block_bootstrap_comparison(test_predictions, candidate, baseline, replicates, block_length, seed):
    df = test_predictions.copy()
    df["ae"] = (df.y_true - df.y_pred).abs()
    # Даты целей — кластеры: все категории и horizons одного месяца остаются вместе.
    monthly = df[df.model.isin([candidate, baseline])].groupby(["target_date", "model"]).ae.mean().unstack()
    difference = (monthly[candidate] - monthly[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(difference)
    samples = []
    for _ in range(replicates):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
        samples.append(difference[indices].mean())
    return {"candidate": candidate, "baseline": baseline, "difference_mae": float(difference.mean()),
            "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975)),
            "n_target_months": n, "block_months": block_length, "replicates": replicates}
