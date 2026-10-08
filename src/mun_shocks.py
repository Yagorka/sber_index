"""Онлайн-детекторы структурных сдвигов на муниципальных остатках.

Остаток r[i,t] = log y[i,t] - log прогноз NatPath_K3 h=1, выпущенный в t-1 (только прошлое).
Масштаб — траектория остатков до t-1, с сжатием к масштабу категории (сжатие по k=6
псевдонаблюдений): на 6–18 точках собственный масштаб ряда слишком шумный.
Детекторы — те же CUSUM, Page-Hinkley и BOCPD из src.transitions.
"""

import numpy as np

from src.mun_models import base_forecast_log
from src.transitions import Detector

FIRST_RESIDUAL_ORIGIN = 5       # origin >= 5 даёт остаток для месяца t = origin + 1 >= 6
MIN_SCALE_HISTORY = 4
SHRINK_K = 6.0
MIN_LOG_SCALE = 0.01


def residual_matrix(ctx):
    """(S, T) остатки log y[t] - forecast(t-1 -> t); NaN для t < 6."""
    p = ctx.panel
    r = np.full((p.n_series, p.n_months), np.nan)
    for t in range(FIRST_RESIDUAL_ORIGIN + 1, p.n_months):
        r[:, t] = p.log[:, t] - base_forecast_log(ctx, t - 1, 1)
    return r


def category_scales(r, categories, first_t=FIRST_RESIDUAL_ORIGIN + 1 + MIN_SCALE_HISTORY):
    """Опорные масштабы категорий по каждому t (только по остаткам < t): (T, n_cat)."""
    S, T = r.shape
    n_cat = int(categories.max()) + 1
    out = np.full((T, n_cat), np.nan)
    for t in range(first_t, T):
        past = r[:, FIRST_RESIDUAL_ORIGIN + 1:t]
        center = np.median(past, axis=1)
        own = 1.4826 * np.median(np.abs(past - center[:, None]), axis=1)
        for c in range(n_cat):
            out[t, c] = np.median(own[categories == c])
    return out


def standardize(r, categories, cat_ref=None, first_t=FIRST_RESIDUAL_ORIGIN + 1 + MIN_SCALE_HISTORY):
    """z[i,t] использует только r[i, <t]. Центр — медиана прошлых остатков ряда.

    cat_ref — опорные масштабы категорий (из неизменённых данных), чтобы при работе
    с подвыборкой/инъекциями масштаб категории не зависел от добавленных сдвигов.
    """
    S, T = r.shape
    if cat_ref is None:
        cat_ref = category_scales(r, categories, first_t)
    z = np.full_like(r, np.nan)
    for t in range(first_t, T):
        past = r[:, FIRST_RESIDUAL_ORIGIN + 1:t]
        n = past.shape[1]
        center = np.median(past, axis=1)
        own = 1.4826 * np.median(np.abs(past - center[:, None]), axis=1)
        scale = np.sqrt((n * own ** 2 + SHRINK_K * cat_ref[t, categories] ** 2) / (n + SHRINK_K))
        z[:, t] = (r[:, t] - center) / np.maximum(scale, MIN_LOG_SCALE)
    return z


def run_detector(z_row, method, threshold, cooldown=3, hazard=1 / 24):
    """Возвращает бинарный массив тревог по месяцам (NaN-входы пропускаются)."""
    det = Detector(method, threshold, cooldown, hazard)
    alarms = np.zeros(len(z_row), dtype=bool)
    k = 0
    for t, v in enumerate(z_row):
        if not np.isfinite(v):
            continue
        _, a = det.update(float(v), k)
        alarms[t] = a and k >= 2
        k += 1
    return alarms


def run_detector_matrix(z, method, threshold, **kw):
    return np.vstack([run_detector(row, method, threshold, **kw) for row in z])


def match_alarms(event_t, alarm_t, max_delay=3):
    """Один-к-одному: первая тревога в [event, event+max_delay] (повторы не растят recall)."""
    for t in alarm_t:
        if event_t <= t <= event_t + max_delay:
            return t
    return None
