"""Муниципальный прогноз 1/3/6/12 мес.: модели, выбор на validation, заморозка, один показ test.

Порядок (важен для честности):
  1. считаются прогнозы всех моделей для всех origin (метки <= origin, признаки <= origin);
  2. по validation выбираются конфигурация структурной модели и веса ансамбля;
  3. frozen_selection.json записывается ДО подсчёта метрик test;
  4. считаются test-метрики, bootstrap, интервалы, прогноз на 2025 г.
  python scripts/run_municipal.py [--recompute]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import CATEGORIES, NationalPriors, load_municipal
from src.mun_ensemble import conformal_quantiles, simplex_lad_weights
from src.mun_eval import (HORIZONS, cluster_bootstrap_diff, make_grid, month_bootstrap_diff,
                          pair_errors, per_series_mae, score_model)
from src.mun_models import (Context, apply_drift, fit_hgb, predict_drift,
                            simple_baselines, training_set)

CACHE = ROOT / "artifacts/municipal_runs/_cache"
N_MONTHS, FIRST_ORIGIN, LAST_ORIGIN = 24, 5, 23
FOUNDATION = {"Chronos2": "chronos2", "Chronos2_NatCov": "chronos2_nat",
              "ChronosBolt": "bolt", "TimesFM": "timesfm"}


def nearest_h(h):
    return min(HORIZONS, key=lambda k: (abs(k - h), k))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def empty(panel):
    return np.full((N_MONTHS, 13, panel.n_series), np.nan, dtype=np.float32)


def cached(name, recompute, builder):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists() and not recompute:
        return np.load(path)
    arr = builder()
    np.save(path, arr)
    return arr


def struct_predictions(ctx, cfg, weight_power):
    """Кэшируем предсказанный дрейф (остаток на месяц); сжатие и cap применяются после."""
    s = cfg["structural"]
    out = empty(ctx.panel)
    for origin in range(FIRST_ORIGIN, LAST_ORIGIN + 1):
        train = training_set(ctx, origin, weight_power, max_rows=s.get("max_train_rows"), seed=cfg["seed"])
        model = None
        if train is not None and len(train[1]) >= s["min_train_rows"]:
            model = fit_hgb(train, s["hgb_params"], cfg["seed"])
        out[origin] = predict_drift(ctx, origin, model)
        print(f"  struct wp={weight_power} nat={ctx.use_nat} origin {origin}", flush=True)
    return out


def load_prophet(panel):
    df = pd.read_parquet(ROOT / "artifacts/municipal_runs/_prophet/prophet_default.parquet")
    out = empty(panel)
    out[df.origin_idx.to_numpy(), df.h.to_numpy(), df.series_idx.to_numpy()] = df.y_pred.to_numpy(dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--precompute-only", action="store_true", help="только кэш структурных прогнозов")
    args = ap.parse_args()
    started = time.monotonic()
    cfg = json.loads((ROOT / "configs/municipal.json").read_text())
    nat_cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    nat = NationalPriors.from_file(ROOT / nat_cfg["input"])
    panel, dropped = load_municipal(ROOT)
    ctx = Context(panel, nat)
    ctx_local = Context(panel, nat, use_nat=False)
    grid = make_grid()
    run_id = datetime.now(timezone.utc).strftime("municipal_%Y%m%dT%H%M%SZ")
    run = ROOT / "artifacts/municipal_runs" / run_id
    if not args.precompute_only:
        run.mkdir(parents=True)
    print(f"RUN {run_id}; series {panel.n_series}; incomplete series dropped {dropped}", flush=True)

    # 1. прогнозы всех моделей -------------------------------------------------------
    preds = {}
    simple = {}
    def build_simple():
        stack = {}
        for o in range(FIRST_ORIGIN, LAST_ORIGIN + 1):
            for name, a in simple_baselines(ctx, o).items():
                stack.setdefault(name, empty(panel))[o] = a
        return stack
    for name in ["LastValue", "SeasonalNaive", "SeasonalNaive_NatGrowth", "NatPath_K1", "NatPath_K3"]:
        preds[name] = None
    stack = None
    if args.recompute or any(not (CACHE / f"{n}.npy").exists() for n in preds):
        stack = build_simple()
        CACHE.mkdir(parents=True, exist_ok=True)
        for n, a in stack.items():
            np.save(CACHE / f"{n}.npy", a)
    for n in list(preds):
        preds[n] = np.load(CACHE / f"{n}.npy")
    preds["Prophet"] = load_prophet(panel)
    struct_full = {}
    for wp in cfg["structural"]["weight_powers"]:
        struct_full[wp] = cached(f"StructDrift_wp{wp}", args.recompute, lambda wp=wp: struct_predictions(ctx, cfg, wp))
    local_full = cached("LocalOnlyDrift_wp0.5", args.recompute, lambda: struct_predictions(ctx_local, cfg, 0.5))
    if args.precompute_only:
        print("кэш готов", flush=True)
        return
    for label, key in FOUNDATION.items():
        path = ROOT / f"artifacts/municipal_runs/_foundation/{key}_median.npy"
        if path.exists():
            preds[label] = np.load(path)
        else:
            print(f"  нет {label}: {path.name} отсутствует — модель пропущена", flush=True)
    base3 = preds["NatPath_K3"]

    # 2. выбор структурной модели на validation ---------------------------------------
    sel_rows = []
    for wp, full in struct_full.items():
        for shrink in cfg["structural"]["shrinks"]:
            cand = apply_drift(base3, full, shrink)
            m = pd.concat([score_model(panel, cand, grid, "validation", "x")])
            sel_rows.append({"weight_power": wp, "shrink": shrink, "validation_mae_mean_h": m.mae.mean(),
                             **{f"mae_h{int(r.h)}": r.mae for r in m.itertuples()}})
    sel = pd.DataFrame(sel_rows).sort_values("validation_mae_mean_h")
    sel.to_csv(run / "structural_selection_validation.csv", index=False)
    best = sel.iloc[0]
    preds["StructHGB"] = apply_drift(base3, struct_full[best.weight_power], best.shrink)
    # LocalOnly: остаток относительно локального сглаженного уровня (без национальных приоров)
    local_base = empty(panel)
    for o in range(FIRST_ORIGIN, LAST_ORIGIN + 1):
        lvl = np.mean([panel.log[:, o - j] for j in range(min(3, o + 1))], axis=0)
        local_base[o, 1:] = np.exp(lvl)[None, :]
    preds["LocalOnlyHGB"] = apply_drift(local_base, local_full, best.shrink)
    preds["LocalBase_K3"] = local_base
    print("Выбрана структурная модель:", dict(best), flush=True)

    # 3. ансамбль: веса по validation и заморозка -----------------------------------------
    candidates = [c for c in cfg["ensemble"]["candidates"] if c in preds]
    gv, yv, _, _ = pair_errors(panel, preds["Prophet"], grid, "validation")
    stackv = {c: pair_errors(panel, preds[c], grid, "validation")[2] for c in candidates}
    weights = {}
    for h in HORIZONS:
        use_h = [h] if h != 12 else [6, 12]
        rows_y, rows_p = [], []
        for hh in use_h:
            m = (gv.h == hh).to_numpy()
            rows_y.append(yv[m].ravel())
            rows_p.append(np.column_stack([stackv[c][m].ravel() for c in candidates]))
        w = simplex_lad_weights(np.concatenate(rows_y), np.vstack(rows_p), seed=cfg["seed"])
        weights[h] = dict(zip(candidates, map(float, w)))
    ens = empty(panel)
    for h in range(1, 13):
        hw = nearest_h(h)   # промежуточные горизонты в протоколе не оцениваются: веса ближайшего оценочного
        ens[:, h] = sum(weights[hw][c] * preds[c][:, h] for c in candidates)
    preds["Ensemble"] = ens
    # conformal-квантили по validation
    gq, yq, pq, _ = pair_errors(panel, ens, grid, "validation")
    cat = np.broadcast_to(panel.category_code, yq.shape)
    hz = np.broadcast_to(gq.h.to_numpy()[:, None], yq.shape)
    quant = conformal_quantiles((np.abs(yq - pq) / pq).ravel(), cat.ravel(), hz.ravel(), cfg["intervals"]["coverage"])
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
              "structural": {"weight_power": float(best.weight_power), "shrink": float(best.shrink)},
              "ensemble_candidates": candidates, "ensemble_weights": {str(h): w for h, w in weights.items()},
              "conformal_relative_quantiles": {f"h{h}_{CATEGORIES[c]}": q for (h, c), q in quant.items()},
              "config_sha256": sha256(ROOT / "configs/municipal.json"),
              "test_metrics_computed_after_this_file": True}
    (run / "frozen_selection.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=1))
    pd.DataFrame([{"h": h, "model": c, "weight": w} for h, ws in weights.items() for c, w in ws.items()]
                 ).to_csv(run / "ensemble_weights.csv", index=False)

    # 4. метрики ------------------------------------------------------------------------
    for name, arr in preds.items():
        np.save(run / f"pred_{name}.npy", arr.astype(np.float32))
    names = list(preds)
    val = pd.concat([score_model(panel, preds[n], grid, "validation", n) for n in names])
    test = pd.concat([score_model(panel, preds[n], grid, "test", n) for n in names])
    val.to_csv(run / "metrics_validation.csv", index=False)
    test.to_csv(run / "metrics_test.csv", index=False)
    pivot = test.pivot(index="model", columns="h", values="mae")
    pivot["mean_h"] = pivot.mean(1)
    pivot = pivot.sort_values("mean_h")
    print(pivot.round(1).to_string(), flush=True)
    base = test[test.model == "Prophet"].set_index("h").mae
    gains = test.assign(gain_vs_prophet_pct=lambda d: 100 * (d.h.map(base) - d.mae) / d.h.map(base))
    gains.to_csv(run / "gains_vs_prophet_test.csv", index=False)

    # bootstrap vs Prophet для финальных моделей
    clusters = panel.meta.territory_id.to_numpy()
    boots = []
    for model in [n for n in ["Ensemble", "StructHGB", "NatPath_K3", "Chronos2", "Chronos2_NatCov", "TimesFM", "ChronosBolt"] if n in preds]:
        for h in HORIZONS:
            a = per_series_mae(panel, preds[model], grid, "test", h)
            b = per_series_mae(panel, preds["Prophet"], grid, "test", h)
            d, lo, hi = cluster_bootstrap_diff(a, b, clusters, cfg["bootstrap"]["replicates"], cfg["seed"])
            md, mlo, mhi, nm = month_bootstrap_diff(panel, preds[model], preds["Prophet"], grid, "test", h,
                                                    cfg["bootstrap"]["replicates"], cfg["seed"])
            boots.append({"model": model, "h": h, "mae_diff_vs_prophet": d, "mo_ci_low": lo, "mo_ci_high": hi,
                          "month_ci_low": mlo, "month_ci_high": mhi, "n_target_months": nm})
    pd.DataFrame(boots).to_csv(run / "bootstrap_vs_prophet_test.csv", index=False)

    # по категориям и размеру (терцили среднего уровня в категории по данным до первого origin test)
    level = panel.values[:, :12].mean(1)
    size = pd.Series(level).groupby(panel.category_code).transform(lambda s: pd.qcut(s, 3, labels=False)).to_numpy()
    cat_rows = []
    for model in [n for n in ["Prophet", "Ensemble", "StructHGB", "NatPath_K3", "Chronos2"] if n in preds]:
        g, y, p, _ = pair_errors(panel, preds[model], grid, "test")
        err = np.abs(y - p)
        for h in HORIZONS:
            m = (g.h == h).to_numpy()
            e = err[m].mean(0)
            for c, cn in enumerate(CATEGORIES):
                cat_rows.append({"model": model, "h": h, "group": "category", "value": cn,
                                 "mae": float(e[panel.category_code == c].mean())})
            for q in range(3):
                cat_rows.append({"model": model, "h": h, "group": "size_tercile", "value": f"T{q + 1}",
                                 "mae": float(e[size == q].mean())})
    pd.DataFrame(cat_rows).to_csv(run / "metrics_by_group_test.csv", index=False)

    # интервалы
    g, y, p, _ = pair_errors(panel, ens, grid, "test")
    cats = np.broadcast_to(panel.category_code, y.shape)
    int_rows = []
    for h in HORIZONS:
        m = (g.h == h).to_numpy()
        q = np.vectorize(lambda c: quant[(h, int(c))])(cats[m])
        lo, hi = p[m] * (1 - q), p[m] * (1 + q)
        inside = (y[m] >= lo) & (y[m] <= hi)
        int_rows.append({"h": h, "nominal": cfg["intervals"]["coverage"], "empirical_coverage_test": float(inside.mean()),
                         "mean_relative_half_width": float(q.mean())})
    pd.DataFrame(int_rows).to_csv(run / "intervals_test.csv", index=False)

    # 5. прогноз на 2025 (origin = 2024-12; факты 2025 на уровне МО недоступны) -------------
    rows = []
    months25 = pd.date_range("2025-01-01", periods=12, freq="MS")
    for h in range(1, 13):
        pred = preds["Ensemble"][LAST_ORIGIN, h]
        rows.append(pd.DataFrame({"territory_id": panel.meta.territory_id, "category": panel.meta.category,
                                  "target_month": months25[h - 1], "horizon_months": h,
                                  "y_pred_ensemble": pred,
                                  "y_pred_structural": preds["StructHGB"][LAST_ORIGIN, h],
                                  "y_pred_prophet": preds["Prophet"][LAST_ORIGIN, h]}))
    f25 = pd.concat(rows, ignore_index=True)
    code = {c: i for i, c in enumerate(CATEGORIES)}
    qrel = np.array([quant[(nearest_h(int(h)), code[c])] for h, c in zip(f25.horizon_months, f25.category)])
    f25["lo90"], f25["hi90"] = f25.y_pred_ensemble * (1 - qrel), f25.y_pred_ensemble * (1 + qrel)
    f25.to_csv(run / "forecast_2025.csv", index=False)

    # 6. журнал и манифест --------------------------------------------------------------------
    journal = test.copy()
    journal["run_id"] = run_id
    path = ROOT / "tracking/municipal_forecast_metrics.csv"
    journal.to_csv(path, mode="a", header=not path.exists(), index=False)
    manifest = {"run_id": run_id, "elapsed_seconds": time.monotonic() - started,
                "data_sha256": sha256(ROOT / "data/raw/consumption.parquet"),
                "config_sha256": sha256(ROOT / "configs/municipal.json"),
                "models": names, "series": panel.n_series, "incomplete_series_dropped": dropped,
                "protocol_assumption": cfg["protocol"]["assumption"],
                "selected_structural": dict(weight_power=float(best.weight_power), shrink=float(best.shrink))}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (ROOT / "artifacts/municipal_runs/latest.json").write_text(json.dumps({"run_id": run_id, "directory": f"artifacts/municipal_runs/{run_id}"}))
    print("готово", run)


if __name__ == "__main__":
    main()
