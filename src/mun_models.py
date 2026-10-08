"""Модели муниципального прогноза. Все признаки в якоре a используют данные <= a.

Базовая идея «структурной» модели: коротких муниципальных рядов (6–24 мес.) не хватает,
чтобы оценить сезонность и рост, поэтому берём их из национальных рядов (с 2018 г.),
а по МО обучаем только отклонение от национального пути.
    прогноз = exp( уровень_МО(a) + национальный путь(a→a+h) + r_hat )
"""

import numpy as np
import pandas as pd

from src.mun_data import CATEGORIES, NATIONAL_MAP, NATIONAL_COLUMNS

LEVEL_K = 3  # число последних месяцев в сглаженной оценке уровня


class Context:
    def __init__(self, panel, nat, use_nat=True):
        # use_nat=False — абляция: национальные пути, рост и реализованные изменения обнуляются.
        self.panel, self.nat, self.use_nat = panel, nat, use_nat
        meta = panel.meta
        self.pick = np.array([nat.col[NATIONAL_MAP[c]] for c in meta.category])
        self.cat = panel.category_code
        # индекс строки «Все категории» того же МО
        key = {(t, c): i for i, (t, c) in enumerate(zip(meta.territory_id, meta.category))}
        self.all_row = np.array([key[(t, "Все категории")] for t in meta.territory_id])
        self.region = meta.region_code.to_numpy()
        self.nat_idx = np.array([nat.idx(d) for d in panel.months])
        self.static = np.column_stack([
            meta.lat.to_numpy(float), meta.lon.to_numpy(float),
            np.log(meta.market_access.to_numpy(float).clip(1e-3)), meta.mo_type_code.to_numpy(float)])
        self._cache = {}
        self.extra = None   # необязательный провайдер внешних признаков: extra(a) -> (S, k)

    # --- национальные приоры, mapped на ряды -----------------------------------------
    def _zeros(self):
        return np.zeros(self.panel.n_series)

    def nat_path(self, a, h, damping=1.0):
        if not self.use_nat:
            return self._zeros()
        return self.nat.path(self.panel.months[a], h, damping)[self.pick]

    def nat_growth(self, a):
        if not self.use_nat:
            return self._zeros()
        return self.nat.growth(self.panel.months[a])[self.pick]

    def nat_realized(self, a, k):
        if not self.use_nat:
            return self._zeros()
        return self.nat.realized(self.panel.months[a], k)[self.pick]

    def level(self, a, k=LEVEL_K, upto_window=True):
        """Национально-скорректированный уровень в месяце a: среднее по последним k лагам."""
        L = self.panel.log
        parts = [L[:, a - j] + (self.nat_realized(a, j) if j else 0.) for j in range(min(k, a + 1))]
        return np.mean(parts, axis=0)


def base_forecast_log(ctx, a, h, k=LEVEL_K, damping=1.0):
    return ctx.level(a, k) + ctx.nat_path(a, h, damping)


def simple_baselines(ctx, o):
    """Словарь name -> (13, S) прогнозов (индекс по h; h=0 не используется)."""
    p = ctx.panel
    out = {n: np.full((13, p.n_series), np.nan) for n in
           ["LastValue", "SeasonalNaive", "SeasonalNaive_NatGrowth", "NatPath_K1", "NatPath_K3"]}
    for h in range(1, 13):
        out["LastValue"][h] = p.values[:, o]
        ref = o + h - 12
        if ref >= 0:
            out["SeasonalNaive"][h] = p.values[:, ref]
            out["SeasonalNaive_NatGrowth"][h] = p.values[:, ref] * np.exp(ctx.nat_growth(o))
        else:
            out["SeasonalNaive"][h] = p.values[:, o]          # fallback = last value (честно в README)
            out["SeasonalNaive_NatGrowth"][h] = p.values[:, o]
        out["NatPath_K1"][h] = np.exp(base_forecast_log(ctx, o, h, 1))
        out["NatPath_K3"][h] = np.exp(base_forecast_log(ctx, o, h, 3))
    return out


# --- признаки для остаточной (GBM/Ridge) модели ---------------------------------------
FEATURES = (["h", "cat", "pc", "pc_seasonal", "log_g", "e1", "e2", "e3", "e5", "e12", "noise",
             "rel_level", "share_all", "ls_gap", "ls", "sin_t", "cos_t", "cs_e3", "cs_e1",
             "reg_e3", "reg_e1", "a_idx", "lat", "lon", "log_access", "mo_type", "mean_level"])
CATEGORICAL = ["cat"]


def _group_mean(values, groups, n_groups):
    s = np.bincount(groups, weights=np.nan_to_num(values), minlength=n_groups)
    c = np.bincount(groups, weights=np.isfinite(values).astype(float), minlength=n_groups)
    return s / np.maximum(c, 1)


def anchor_features(ctx, a):
    """Признаки, не зависящие от горизонта (кэшируются)."""
    if a in ctx._cache:
        return ctx._cache[a]
    L = ctx.panel.log
    S = ctx.panel.n_series
    f = {}
    for k in (1, 2, 3, 5, 12):
        if a - k >= 0:
            f[f"e{k}"] = (L[:, a] - L[:, a - k]) - ctx.nat_realized(a, k)
        else:
            f[f"e{k}"] = np.full(S, np.nan)
    lo = max(1, a - 7)
    adj = []
    for t in range(lo, a + 1):
        step = (ctx.nat.log[ctx.nat_idx[t]] - ctx.nat.log[ctx.nat_idx[t - 1]])[ctx.pick] if ctx.use_nat else 0.
        adj.append((L[:, t] - L[:, t - 1]) - step)
    f["noise"] = np.std(np.stack(adj), axis=0) if len(adj) >= 2 else np.full(S, np.nan)
    med = pd.Series(L[:, a]).groupby(ctx.cat).transform("median").to_numpy()
    f["rel_level"] = L[:, a] - med
    f["share_all"] = L[:, a] - L[ctx.all_row, a]
    f["mean_level"] = L[:, :a + 1].mean(1)
    cat_ids = ctx.cat
    n_cat = len(CATEGORIES)
    for k in (1, 3):
        e = f[f"e{k}"]
        f[f"cs_e{k}"] = _group_mean(e, cat_ids, n_cat)[cat_ids]
        grp = ctx.region * n_cat + cat_ids
        uniq, inv = np.unique(grp, return_inverse=True)
        f[f"reg_e{k}"] = _group_mean(e, inv, len(uniq))[inv]
    f["log_g"] = ctx.nat_growth(a)
    f["lat"], f["lon"], f["log_access"], f["mo_type"] = ctx.static.T
    f["a_idx"] = np.full(S, float(a))
    ctx._cache[a] = f
    return f


def feature_matrix(ctx, a, h):
    f = anchor_features(ctx, a)
    L = ctx.panel.log
    S = ctx.panel.n_series
    pc = ctx.nat_path(a, h)
    seasonal = pc - f["log_g"]
    ref = a + h - 12
    if ref >= 0:
        ls = L[:, ref] - L[:, a]
        gap = ls - seasonal
    else:
        ls = gap = np.full(S, np.nan)
    month = ctx.panel.months[a].month + h
    phase = 2 * np.pi * ((month - 1) % 12) / 12
    cols = {"h": np.full(S, float(h)), "cat": ctx.cat.astype(float), "pc": pc, "pc_seasonal": seasonal,
            "ls": ls, "ls_gap": gap, "sin_t": np.full(S, np.sin(phase)), "cos_t": np.full(S, np.cos(phase))}
    cols.update(f)
    X = np.column_stack([cols[n] for n in FEATURES]).astype(np.float32)
    if ctx.extra is not None:
        X = np.hstack([X, ctx.extra(a).astype(np.float32)])
    base = ctx.level(a, LEVEL_K) + pc
    return X, base


# --- остаточная модель -----------------------------------------------------------------
MIN_TRAIN_ROWS = 3000


def training_set(ctx, o, weight_power=0.5, per_month=True, max_h=12, max_rows=None, seed=42):
    """Все пары (якорь a, горизонт h) с меткой a+h <= o и историей >= 6 мес. (a >= 5)."""
    L = ctx.panel.log
    xs, ys, ws, meta = [], [], [], []
    for a in range(MIN_ORIGIN_FEATURES, o):
        for h in range(1, min(max_h, o - a) + 1):
            X, base = feature_matrix(ctx, a, h)
            r = L[:, a + h] - base
            xs.append(X)
            ys.append(r / h if per_month else r)
            level = np.exp(L[:, a]) ** weight_power
            ws.append((h if per_month else 1.) * level / level.mean())
            meta.append((a, h))
    if not xs:
        return None
    assert max(a + h for a, h in meta) <= o  # метки не позже origin
    X, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    if max_rows and len(y) > max_rows:   # фиксированная случайная подвыборка строк (скорость)
        keep = np.sort(np.random.default_rng(seed + o).choice(len(y), max_rows, replace=False))
        X, y, w = X[keep], y[keep], w[keep]
    return X, y, w, meta


MIN_ORIGIN_FEATURES = 5


def fit_hgb(train, params, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y, w, _ = train
    X = X.copy()
    X[:, np.isnan(X).all(0)] = 0.   # на ранних origin часть признаков (e12, ls) ещё не определена
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[FEATURES.index("cat")] = True   # дополнительные признаки идут после FEATURES, числовые
    model = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=mask,
                                          random_state=seed, early_stopping=False, **params)
    model.fit(X, y, sample_weight=w)
    return model


def predict_drift(ctx, o, model):
    """(13, S) предсказанный остаток в расчёте на месяц (r/h); NaN без модели."""
    out = np.full((13, ctx.panel.n_series), np.nan)
    if model is None:
        return out
    for h in range(1, 13):
        X, _ = feature_matrix(ctx, o, h)
        out[h] = model.predict(X)
    return out


def max_trained_horizon(o):
    """Наибольший горизонт, встречавшийся в обучении на origin o: якорь a >= 5, метка a+h <= o."""
    return max(0, o - MIN_ORIGIN_FEATURES)


def apply_drift(base, drift, shrink):
    """pred = base * exp(shrink * d_hat * min(h, h_trained)); без экстраполяции за обученный горизонт.

    base, drift: (24, 13, S). Если модели не было (NaN) — остаётся база.
    """
    out = base.copy()
    for o in range(base.shape[0]):
        hmax = max_trained_horizon(o)
        for h in range(1, 13):
            d = drift[o, h]
            if hmax == 0 or not np.isfinite(d).any():
                continue
            out[o, h] = base[o, h] * np.exp(shrink * np.nan_to_num(d) * min(h, hmax))
    return out.astype(np.float32)
