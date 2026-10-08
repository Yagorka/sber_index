"""Адаптация использует только созревшие прогнозные ошибки target <= origin."""

import calendar
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.forecasting import direct_features


def calendar_features(date):
    days = pd.date_range(date, periods=calendar.monthrange(date.year, date.month)[1], freq="D")
    # Арифметические будни, без предположений о российском производственном календаре.
    return [len(days), int((days.weekday < 5).sum()), int((days.weekday >= 5).sum())] + [float(date.month == m) for m in range(1, 13)]


def fit_adaptive_catboost(panel, origin, horizons, variant, params, seed):
    output, fitted, audit = [], {}, []
    for h in horizons:
        x, y, labels = [], [], []
        lower = max(23 + h, origin - variant["window"] + 1) if variant["window"] else 23 + h
        for j, series in enumerate(panel.columns):
            values = panel[series].to_numpy()
            for target in range(lower, origin + 1):
                anchor = target - h
                features, reference = direct_features(values, anchor, h, j, len(panel.columns))
                if variant["calendar"]:
                    features = np.r_[features, calendar_features(panel.index[target])]
                x.append(features)
                y.append(values[target] / reference - 1)
                labels.append(target)
        if not labels or max(labels) > origin:
            raise ValueError("Некорректная временная граница обучения")
        age = origin - np.asarray(labels)
        weights = np.power(0.5, age / variant["half_life"]) if variant["half_life"] else np.ones(len(labels))
        model = CatBoostRegressor(**params, loss_function="MAE", random_seed=seed,
                                  thread_count=1, verbose=False, allow_writing_files=False)
        model.fit(np.asarray(x), np.asarray(y), sample_weight=weights)
        fitted[h] = model
        audit.append({"variant": variant["id"], "origin_index": origin, "horizon_months": h,
                      "minimum_label_index": min(labels), "maximum_label_index": max(labels), "training_examples": len(labels)})
        for j, series in enumerate(panel.columns):
            features, reference = direct_features(panel[series].to_numpy(), origin, h, j, len(panel.columns))
            if variant["calendar"]:
                features = np.r_[features, calendar_features(panel.index[origin] + pd.DateOffset(months=h))]
            prediction = reference * (1 + float(model.predict(features.reshape(1, -1))[0]))
            output.append((h, series, max(0., prediction)))
    return output, fitted, audit


def calibration_weights(stream, ids, calibration_range):
    rows = stream[stream.model.isin(ids) & stream.origin_index.between(*calibration_range)].copy()
    rows["ae"] = (rows.y_pred - rows.y_true).abs()
    weights = {}
    for h, part in rows.groupby("horizon_months"):
        errors = part.groupby("model").ae.mean().reindex(ids)
        inverse = 1 / errors.clip(lower=1e-6)
        weights[int(h)] = (inverse / inverse.sum()).to_dict()
    return weights


def mix_stream(stream, ids, name, fixed_weights=None, window=None, regime=None, min_errors=3):
    data = stream[stream.model.isin(ids)].copy()
    data["ae"] = (data.y_pred - data.y_true).abs()
    output, logs = [], []
    for (origin_index, h), current in data.groupby(["origin_index", "horizon_months"], sort=True):
        matured = data[(data.horizon_months == h) & (data.target_index <= origin_index) & data.y_true.notna()]
        if fixed_weights is not None:
            weights = fixed_weights[int(h)]
            used = None
        else:
            lookback = window
            if regime is not None:
                history = regime[regime.index <= origin_index]
                alert = bool(len(history) and history.iloc[-1])
                lookback = 3 if alert else 12
            used = matured[matured.target_index > origin_index - lookback]
            counts = used.groupby("model").target_index.nunique().reindex(ids, fill_value=0)
            if (counts >= min_errors).all():
                errors = used.groupby("model").ae.mean().reindex(ids)
                inverse = 1 / errors.clip(lower=1e-6)
                # Небольшая доля равных весов не даёт одному эксперту вытеснить остальных.
                vector = .8 * inverse / inverse.sum() + .2 / len(ids)
                weights = vector.to_dict()
            else:
                weights = {model: 1 / len(ids) for model in ids}
        for series, group in current.groupby("series_id"):
            indexed = group.set_index("model")
            if set(indexed.index) != set(ids):
                raise ValueError("Неполный состав экспертов")
            row = group.iloc[0].to_dict()
            row["model"] = row["variant"] = name
            row["y_pred"] = sum(indexed.loc[model, "y_pred"] * weights[model] for model in ids)
            output.append(row)
        logs.append({"model": name, "origin_index": origin_index, "horizon_months": h,
                     "maximum_error_target_index": int(used.target_index.max()) if used is not None and len(used) else None,
                     **{f"weight_{m}": weights[m] for m in ids}})
    return pd.DataFrame(output).drop(columns="ae", errors="ignore"), pd.DataFrame(logs)


def correct_bias(stream, name, window, shrinkage=.5, min_errors=3):
    data = stream.copy()
    data["error"] = data.y_pred - data.y_true
    output, logs = [], []
    for row in data.itertuples(index=False):
        history = data[(data.series_id == row.series_id) & (data.horizon_months == row.horizon_months) &
                       (data.target_index <= row.origin_index) & (data.target_index > row.origin_index - window) & data.y_true.notna()]
        # Исходный ряд здесь содержит один метод и один прогноз на target/h/series.
        bias = float(history.error.median()) * shrinkage if len(history) >= min_errors else 0.
        record = row._asdict()
        record.pop("error", None)
        record.update(model=name, variant=name, y_pred=max(0., row.y_pred - bias))
        output.append(record)
        logs.append({"model": name, "origin_index": row.origin_index, "horizon_months": row.horizon_months,
                     "series_id": row.series_id, "bias_correction": bias,
                     "maximum_error_target_index": int(history.target_index.max()) if len(history) else None})
    return pd.DataFrame(output), pd.DataFrame(logs)


def reconcile(stream, name, calibration_range, method="bottom_up", shrinkage=.2):
    categories = sorted(stream.series_id.unique())
    total = categories.index("Всего")
    constraint = -np.ones(len(categories)); constraint[total] = 1
    covariance = {}
    for h, group in stream[stream.origin_index.between(*calibration_range)].groupby("horizon_months"):
        part = group.copy(); part["error"] = part.y_pred - part.y_true
        matrix = part.pivot(index="target_index", columns="series_id", values="error").reindex(columns=categories).to_numpy()
        cov = np.cov(matrix, rowvar=False)
        covariance[h] = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov)) + np.eye(len(categories)) * 1e-6
    output = []
    for (_, h), group in stream.groupby(["origin_index", "horizon_months"]):
        group = group.set_index("series_id").reindex(categories).copy()
        values = group.y_pred.to_numpy().copy()
        if method == "bottom_up":
            values[total] = values[np.arange(len(values)) != total].sum()
        else:
            cov = covariance[h]
            values -= cov @ constraint * (constraint @ values) / (constraint @ cov @ constraint)
            if (values < 0).any():
                values = np.maximum(values, 0)
                values[total] = values[np.arange(len(values)) != total].sum()
        group["y_pred"] = values
        group["model"] = group["variant"] = name
        output.append(group.reset_index())
    return pd.concat(output, ignore_index=True), covariance
