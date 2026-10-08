"""Дополнительные эксперименты после просмотра test: отдельный каталог и манифест."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from src.consumer import read_export
from src.mun_data import MunicipalPanel, NationalPriors, load_municipal
from src.mun_eval import (HORIZONS, make_grid, pair_errors, score_model, per_series_mae,
                         cluster_bootstrap_diff, month_bootstrap_diff)
from src.mun_shocks import residual_matrix, run_detector_matrix
from src.mun_models import Context
from src.research_audit import rolling_blend, past_calibration, shock_adjustment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seasonal_ridge(panel, alpha):
    """Общий регуляризованный сезонный тренд, отдельные коэффициенты на ряд; прошлое."""
    out = np.full((24, 13, panel.n_series), np.nan, dtype=np.float32)
    def features(t):
        return np.column_stack([np.ones(len(t)), t/12,
                                np.sin(2*np.pi*t/12), np.cos(2*np.pi*t/12)])
    for o in range(5, 24):
        X = features(np.arange(o+1))
        penalty = np.diag([0., alpha, alpha, alpha])
        coef = np.linalg.solve(X.T @ X + penalty, X.T @ panel.log[:, :o+1].T)
        out[o, 1:] = np.exp(features(np.arange(o+1, o+13)) @ coef)
    return out


def external_marketplace(panel, base):
    cfg = json.loads((ROOT / "configs/consumer_sources.json").read_text())
    raw, _ = read_export(ROOT / cfg["datasets"]["weekly_growth"]["preferred_file"])
    w = raw[raw.category.str.strip() == "Маркетплейсы"].copy()
    w["period"] = pd.to_datetime(w.period)
    w["available_at"] = w.period + pd.Timedelta(days=7)
    out = base.copy()
    audit = []
    ids = np.where(panel.meta.category == "Маркетплейсы")[0]
    for o in range(5, 24):
        end = panel.months[o] + pd.offsets.MonthEnd(0)
        known = w[(w.available_at <= end) & (w.period > end - pd.Timedelta(weeks=13))]
        growth = known.value.median() if len(known) >= 4 else np.nan
        for h in range(1, 13):
            ref = o+h-12
            applied = np.isfinite(growth) and ref >= 0
            if applied:
                out[o, h, ids] = panel.values[ids, ref] * max(0., 1+growth/100)
            audit.append({"origin": o, "h": h, "weeks": len(known), "applied": applied,
                          "last_available_at": known.available_at.max(), "yoy_pct": growth,
                          "fallback": "SeasonalNaive_NatGrowth" if not applied else "none"})
    return out, pd.DataFrame(audit), w


def main():
    cfg = json.loads((ROOT / "configs/research_audit.json").read_text())
    source = ROOT / cfg["source_run"]
    if not (source / "pred_Ensemble.npy").exists():
        # На чистом checkout тяжёлые массивы исходного run не хранятся в git.
        # После make all используется только что воспроизведённый муниципальный run.
        latest = ROOT / json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())["directory"]
        if not (latest / "pred_Ensemble.npy").exists():
            raise FileNotFoundError("Нет муниципальных прогнозов: сначала выполните make foundation municipal")
        source = latest
    panel, dropped = load_municipal(ROOT)
    grid = make_grid()
    runid = datetime.now(timezone.utc).strftime("research_%Y%m%dT%H%M%S_%fZ")
    run = ROOT / "artifacts/research_runs" / runid
    run.mkdir(parents=True)
    preds = {n: np.load(source / f"pred_{n}.npy") for n in
             ("Prophet", "Ensemble", "SeasonalNaive_NatGrowth", "NatPath_K1", "NatPath_K3", "StructHGB")}
    print("Исходные прогнозы загружены; сравнение и подбор весов", flush=True)
    # Исходные числа воспроизводятся по сохранённым прогнозам, исходные артефакты не меняются.
    old_metrics = pd.read_csv(source / "metrics_test.csv")
    reproduced = pd.concat([score_model(panel, p, grid, "test", n) for n, p in preds.items()])
    check = reproduced.merge(old_metrics, on=["model", "h"], suffixes=("_recomputed", "_saved"))
    check["mae_abs_difference"] = (check.mae_recomputed - check.mae_saved).abs()
    check[["model", "h", "mae_abs_difference"]].to_csv(run / "reproduction.csv", index=False)
    if check.mae_abs_difference.max() > 1e-5:
        raise ValueError("Сохранённые метрики не воспроизводятся")
    # Равные веса не требуют подбора. Вариант без HGB подбирается ретроспективно на dev.
    simple_names = ["Prophet", "SeasonalNaive_NatGrowth", "NatPath_K3"]
    preds["EqualBlend_NoHGB"] = sum(preds[n] for n in simple_names)/len(simple_names)
    from src.mun_ensemble import simplex_lad_weights
    no_hgb = np.full_like(preds["Prophet"], np.nan)
    weights = []
    gv, yv, _, _ = pair_errors(panel, preds["Prophet"], grid, "validation")
    for h in HORIZONS:
        mask = gv.h.isin([6, 12] if h == 12 else [h]).to_numpy()
        P = np.column_stack([pair_errors(panel, preds[n], grid, "validation")[2][mask].ravel() for n in simple_names])
        w = simplex_lad_weights(yv[mask].ravel(), P, seed=cfg["seed"], max_rows=cfg["weight_fit_max_rows"])
        no_hgb[:, h] = sum(v*preds[n][:, h] for n, v in zip(simple_names, w))
        weights.extend({"h": h, "model": n, "weight": v} for n, v in zip(simple_names, w))
    preds["DevBlend_NoHGB"] = no_hgb
    pd.DataFrame(weights).to_csv(run / "no_hgb_weights.csv", index=False)
    ridge_variants = {f"RidgeSeasonal_alpha{a}": seasonal_ridge(panel, a) for a in (0.1, 1., 10.)}
    dev = pd.concat([score_model(panel, arr, grid, "validation", n) for n, arr in ridge_variants.items()])
    selected = dev.groupby("model").mae.mean().idxmin()
    preds["RidgeSeasonal_DevSelected"] = ridge_variants[selected]
    dev.to_csv(run / "ridge_selection.csv", index=False)
    rolling, audit = rolling_blend(panel.values, {n: preds[n] for n in simple_names},
            calibration_months=cfg["calibration_months"], minimum_months=cfg["minimum_selection_months"], seed=cfg["seed"],
            max_rows=cfg["weight_fit_max_rows"])
    print("Последовательные веса готовы; интервалы и внешние данные", flush=True)
    preds["PastOnlyBlend"] = rolling
    audit = pd.DataFrame(audit)
    assert (audit.selection_last_target <= audit.origin-cfg["calibration_months"]).all()
    audit.to_csv(run / "past_only_selection_audit.csv", index=False)
    # Последовательная калибровка: последние два зрелых месяца, не используемые в весах.
    interval_rows, interval_audit = [], []
    for row in grid[grid.stage == "test"].itertuples():
        q, pairs = past_calibration(panel.values, rolling, panel.category_code, row.origin, row.h,
                    cfg["calibration_months"], cfg["minimum_calibration_months"])
        pred, y = rolling[row.origin, row.h], panel.values[:, row.target]
        ok = np.isfinite(q)
        inside = (y >= np.maximum(0., pred*(1-q))) & (y <= pred*(1+q))
        interval_audit.append({"origin": row.origin, "h": row.h, "target": row.target,
                               "calibration_targets": ";".join(str(t) for _, t in pairs),
                               "last_calibration_target": max((t for _, t in pairs), default=-1),
                               "calibration_months": len(pairs), "availability": ok.mean()})
        for cat in panel.meta.category.unique():
            m = (panel.meta.category == cat).to_numpy()
            valid = m & ok
            interval_rows.append({"h": row.h, "origin": row.origin, "target_month": panel.months[row.target],
                      "category": cat, "nominal": .9, "availability": ok[m].mean(),
                      "coverage": inside[valid].mean() if valid.any() else np.nan,
                      "relative_half_width": q[valid].mean() if valid.any() else np.nan,
                      "mean_width_rub": (2*pred[valid]*q[valid]).mean() if valid.any() else np.nan})
    pd.DataFrame(interval_rows).to_csv(run / "past_only_intervals.csv", index=False)
    pd.DataFrame(interval_audit).to_csv(run / "calibration_audit.csv", index=False)
    # Аудит исходного ретроспективного выбора, включая unavailable-before-origin.
    original = grid.copy()
    original["selection_last_target"] = 17
    original["selection_available_at_origin"] = original.origin >= 17
    original.to_csv(run / "original_selection_time_audit.csv", index=False)
    market, market_audit, weekly = external_marketplace(panel, preds["SeasonalNaive_NatGrowth"])
    preds["WeeklyMarketplacePrior"] = market
    market_audit.to_csv(run / "marketplace_availability.csv", index=False)
    # Отдельно национальная сезонность и рост; доля каждого вклада.
    nat = NationalPriors.from_file(ROOT / json.loads((ROOT / "configs/national_forecast.json").read_text())["input"])
    ctx = Context(panel, nat)
    seasonal = np.full_like(preds["NatPath_K3"], np.nan)
    growth = seasonal.copy()
    for o in range(5, 24):
        for h in range(1, 13):
            logg = ctx.nat_growth(o)
            seasonal[o,h] = np.exp(ctx.level(o) + ctx.nat_path(o,h) - logg)
            growth[o,h] = np.exp(ctx.level(o) + logg*h/12)
    preds["NatSeasonalityOnly"] = seasonal
    preds["NatGrowthOnly"] = growth
    # Простая пороговая база: бюджет тревог на той же калибровке 10..17.
    z = np.load(ROOT / "artifacts/shocks/z.npy")
    residuals = residual_matrix(ctx)
    thresholds = np.arange(1., 10.01, .25)
    def threshold_alarms(threshold):
        alarms = np.zeros_like(z, dtype=bool)
        last = np.full(panel.n_series, -100)
        for t in range(z.shape[1]):
            hit = (np.abs(z[:, t]) >= threshold) & (t-last >= 3)
            alarms[:,t] = hit
            last[hit] = t
        return alarms
    for threshold in thresholds:
        simple_alarm = threshold_alarms(threshold)
        if simple_alarm[:,10:18].mean()*100 <= 1:
            break
    np.save(run / "alarms_AbsZ.npy", simple_alarm)
    action_rows = []
    for method in ["AbsZ", "CUSUM", "PageHinkley", "BOCPD"]:
        alarms = simple_alarm if method == "AbsZ" else np.load(ROOT / f"artifacts/shocks/alarms_{method}.npy")
        adjusted = shock_adjustment(preds["NatPath_K3"], residuals, alarms,
                      cfg["shock_action"]["shrink"], cfg["shock_action"]["max_log_correction"], cfg["shock_action"]["decay_months"])
        name = f"ShockAdjusted_{method}"
        preds[name] = adjusted
        for h in HORIZONS:
            g,y,p,_ = pair_errors(panel, adjusted, grid, "test")
            _,_,b,_ = pair_errors(panel, preds["NatPath_K3"], grid, "test")
            m = (g.h == h).to_numpy()
            action_rows.append({"method": method, "h": h, "base_mae": np.abs(y[m]-b[m]).mean(),
                  "adjusted_mae": np.abs(y[m]-p[m]).mean(),
                  "delta_mae": (np.abs(y[m]-p[m])-np.abs(y[m]-b[m])).mean(),
                  "alarm_rate_test_per_100": alarms[:,18:24].mean()*100})
    pd.DataFrame(action_rows).to_csv(run / "shock_action.csv", index=False)
    print("Связка детектор–прогноз готова; метрики и bootstrap", flush=True)
    # Все модели на одной полной сетке. Метрики по категориям, месяцам, регионам и МО.
    metrics = pd.concat([score_model(panel, arr, grid, stage, n) for n, arr in preds.items()
                         for stage in ("validation", "test")])
    metrics.to_csv(run / "metrics.csv", index=False)
    boot, groups, shares = [], [], []
    for name in ["Ensemble", "DevBlend_NoHGB", "EqualBlend_NoHGB", "PastOnlyBlend", "WeeklyMarketplacePrior", "RidgeSeasonal_DevSelected"]:
        for h in HORIZONS:
            a = per_series_mae(panel, preds[name], grid, "test", h)
            for comparator in ["Prophet", "SeasonalNaive_NatGrowth"]:
                b = per_series_mae(panel, preds[comparator], grid, "test", h)
                d,lo,hi = cluster_bootstrap_diff(a,b,panel.meta.territory_id, cfg["bootstrap_replicates"])
                _,ml,mh,nm = month_bootstrap_diff(panel,preds[name],preds[comparator],grid,"test",h,cfg["bootstrap_replicates"])
                boot.append({"model": name, "comparator": comparator, "h": h, "mae_difference": d,
                             "mo_ci_low": lo, "mo_ci_high": hi, "month_ci_low": ml, "month_ci_high": mh,
                             "n_target_months": nm})
                bymo = pd.DataFrame({"territory_id": panel.meta.territory_id, "a": a,"b": b}).groupby("territory_id")[["a","b"]].mean()
                shares.append({"model": name,"comparator": comparator,"h":h,"municipal_win_share": (bymo.a < bymo.b).mean(),
                               "series_win_share": (a<b).mean(),"median_series_relative_gain": np.median((b-a)/np.maximum(b,1e-9))})
        g,y,p,_ = pair_errors(panel,preds[name],grid,"test")
        for rowidx,row in enumerate(g.itertuples()):
            for group, labels in [("category",panel.meta.category), ("region",panel.meta.region_name)]:
                for label in labels.dropna().unique():
                    m = (labels == label).to_numpy()
                    err = np.abs(y[rowidx,m]-p[rowidx,m])
                    groups.append({"model":name,"h":row.h,"target_month":panel.months[row.target],"group":group,"value":label,
                                   "n_series":m.sum(),"mae":err.mean(),"wape":err.sum()/y[rowidx,m].sum(),
                                   "median_ape":np.median(err/y[rowidx,m])})
    pd.DataFrame(boot).to_csv(run / "paired_comparisons.csv",index=False)
    pd.DataFrame(groups).to_csv(run / "group_metrics.csv",index=False)
    pd.DataFrame(shares).to_csv(run / "win_shares.csv",index=False)
    # Сезонный Prophet только на той же подвыборке: нельзя переносить результат на все МО.
    prophet_dir = ROOT / "artifacts/research_prophet"
    if (prophet_dir / "manifest.json").exists():
        pm = json.loads((prophet_dir / "manifest.json").read_text())
        ids = np.where(panel.meta.territory_id.isin(pm["sample_territories"]))[0]
        subset = MunicipalPanel(panel.values[ids],panel.meta.iloc[ids],panel.months)
        subset_preds = {n:arr[:,:,ids] for n,arr in preds.items() if n in ["Prophet","Ensemble","SeasonalNaive_NatGrowth","PastOnlyBlend","RidgeSeasonal_DevSelected"]}
        subset_preds["ProphetSeasonal_DevSelected"] = np.load(prophet_dir / f'{pm["selected_on_validation"]}.npy')[:,:,ids]
        pd.concat([score_model(subset,arr,grid,stage,n) for n,arr in subset_preds.items()
                   for stage in ("validation","test")]).to_csv(run / "prophet_same_subset.csv",index=False)
    # Номинальный рост панели и национальных категорий: одинаковые месяцы, разные веса.
    economic = []
    for cat in panel.meta.category.unique():
        ids = np.where(panel.meta.category == cat)[0]
        for t in range(12,24):
            local = panel.values[ids,t]/panel.values[ids,t-12]-1
            national = np.exp(ctx.nat_realized(t,12))[ids[0]]-1
            economic.append({"category":cat,"month":panel.months[t],"mean_local_yoy_pct":100*local.mean(),
                       "median_local_yoy_pct":100*np.median(local),"ratio_of_sum_yoy_pct":100*(panel.values[ids,t].sum()/panel.values[ids,t-12].sum()-1),
                       "national_proxy_yoy_pct":100*national})
    pd.DataFrame(economic).to_csv(run / "economic_diagnostics.csv",index=False)
    for n in ["PastOnlyBlend", "DevBlend_NoHGB", "RidgeSeasonal_DevSelected", "WeeklyMarketplacePrior"]:
        np.save(run / f"pred_{n}.npy", preds[n].astype(np.float32))
    manifest = {"run_id":runid,"created_at_utc":datetime.now(timezone.utc).isoformat(), "source_run":str(source.relative_to(ROOT)),
                "status":cfg["evaluation_status"],"config":cfg,"data_sha256":digest(ROOT/"data/inputs/municipal_consumption.parquet"),
                "config_sha256":digest(ROOT/"configs/research_audit.json"),"ridge_selected":selected,
                "abs_z_threshold":float(threshold),"incomplete_series_dropped":dropped,
                "weekly_marketplace_first_period":str(weekly.period.min().date()),
                "input_sha256":{str(path.relative_to(ROOT)):digest(path) for path in [
                    ROOT/"data/inputs/municipal_consumption.parquet", ROOT/"configs/research_audit.json",
                    ROOT/json.loads((ROOT/"configs/national_forecast.json").read_text())["input"],
                    ROOT/json.loads((ROOT/"configs/consumer_sources.json").read_text())["datasets"]["weekly_growth"]["preferred_file"],
                    ROOT/"data/inputs/verified_event_cases.csv",ROOT/"data/inputs/cpi_december_2024.csv",
                    ROOT/"requirements.lock.txt"]},
                "limitations":["latest vintage; публикационные задержки месячных данных неизвестны",
                   "полные ряды выбраны по всей истории; complete-cohort evaluation",
                   "короткая история; h=12 as-of веса фиксированы, интервалы недоступны",
                   "официальный baseline и балльная шкала не подтверждены"]}
    (run/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    root = ROOT / "artifacts/research_runs"
    (root/"latest.json").write_text(json.dumps({"directory":str(run.relative_to(ROOT))}))
    print(metrics[metrics.stage=="test"].pivot(index="model",columns="h",values="mae").round(1).to_string())
    print("Research audit:",run)


if __name__ == "__main__":
    main()
