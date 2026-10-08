"""Веса ансамбля (минимизация MAE на симплексе) и split-conformal интервалы."""

import numpy as np
import warnings
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning


def simplex_lad_weights(y, P, max_rows=40000, seed=42):
    """min_w sum |y - P w|, w >= 0, sum w = 1 (линейная программа, HiGHS).

    y: (n,), P: (n, K). Для скорости используется случайная подвыборка строк
    с фиксированным seed; строки с неконечными прогнозами исключаются.
    """
    ok = np.isfinite(P).all(1) & np.isfinite(y)
    y, P = y[ok], P[ok]
    if len(y) > max_rows:
        pick = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        y, P = y[pick], P[pick]
    n, k = P.shape
    # переменные: w (k), u (n) >= |y - P w|
    c = np.r_[np.zeros(k), np.ones(n) / n]
    from scipy.sparse import hstack, vstack, csr_matrix, eye
    Ps = csr_matrix(P)
    I = eye(n, format="csr")
    A = vstack([hstack([Ps, -I]), hstack([-Ps, -I])])
    b = np.r_[y, -y]
    # SciPy передаёт threads в HiGHS; один поток бережёт интерактивную систему.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads", category=OptimizeWarning)
        res = linprog(c, A_ub=A, b_ub=b, A_eq=np.r_[np.ones(k), np.zeros(n)][None, :], b_eq=[1.0],
                      bounds=[(0, None)] * k + [(0, None)] * n, method="highs", options={"threads": 1})
    if res.status != 0:
        raise RuntimeError(res.message)
    w = np.clip(res.x[:k], 0, None)
    return w / w.sum()


def conformal_quantiles(rel_errors, categories, horizons, coverage=0.9):
    """Квантили относительной абсолютной ошибки |y - yhat| / yhat по (h, категория)."""
    out = {}
    for h in np.unique(horizons):
        for c in np.unique(categories):
            m = (horizons == h) & (categories == c)
            e = rel_errors[m]
            e = e[np.isfinite(e)]
            n = len(e)
            level = min(1.0, np.ceil((n + 1) * coverage) / n) if n else 1.0
            out[(int(h), int(c))] = float(np.quantile(e, level)) if n else np.nan
    return out
