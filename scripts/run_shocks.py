"""Детекторы сдвигов и раннее предупреждение на муниципальной панели.

Три независимых способа оценки (синтетика не выдаётся за реальные шоки):
  A. semi-synthetic injection: известные сдвиги добавляются в реальные остатки 5% рядов;
     метрики event precision/recall/F1, задержка, ложные тревоги на 100 ряд-месяцев;
  B. реестр датированных событий (data/inputs/event_registry.csv): совпадение тревог с
     реальными событиями у сопоставленных МО против базовой частоты у остальных;
  C. offline-proxy разметка (оптимальная сегментация по полным данным) — только чувствительность.
Затем раннее предупреждение: вероятность сдвига в (t, t+k] для k=1,3 по proxy-меткам.
Пороги выбираются на калибровочных месяцах (10..17) при бюджете ложных тревог; test — 18..23.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import CATEGORIES, NationalPriors, load_municipal
from src.mun_models import Context
from src.mun_shocks import (category_scales, match_alarms, residual_matrix, run_detector,
                            run_detector_matrix, standardize)
from src.transitions import offline_mean_changes

METHODS = ["CUSUM", "PageHinkley", "BOCPD"]


def alarm_rate(alarms, months):
    lo, hi = months
    return 100 * alarms[:, lo:hi + 1].mean()


def prf(tp, n_alarms, n_events):
    p = tp / n_alarms if n_alarms else 0.
    r = tp / n_events if n_events else 0.
    return p, r, (2 * p * r / (p + r) if p + r else 0.)


def main():
    t0 = time.time()
    cfg = json.loads((ROOT / "configs/shocks.json").read_text())
    nat_cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    nat = NationalPriors.from_file(ROOT / nat_cfg["input"])
    panel, _ = load_municipal(ROOT)
    ctx = Context(panel, nat)
    cats = panel.category_code
    r = residual_matrix(ctx)
    cat_ref = category_scales(r, cats)
    z = standardize(r, cats, cat_ref)
    cal, test = cfg["calibration_months"], cfg["test_months"]
    out = ROOT / "artifacts/shocks"
    out.mkdir(parents=True, exist_ok=True)
    print(f"z готов: {np.isfinite(z).sum()} значений; месяцы с z: {np.where(np.isfinite(z).any(0))[0][[0, -1]]}", flush=True)

    # --- выбор порогов на калибровке при бюджете ложных тревог --------------------------------
    tuning, chosen, base_alarms = [], {}, {}
    for method in METHODS:
        feasible = None
        for thr in cfg["thresholds"][method]:
            a = run_detector_matrix(z, method, thr, cooldown=cfg["cooldown_months"], hazard=cfg["bocpd_hazard"])
            rate = alarm_rate(a, cal)
            tuning.append({"method": method, "threshold": thr, "alarm_rate_calibration_per_100": rate,
                           "alarm_rate_test_per_100": alarm_rate(a, test),
                           "within_budget": rate <= cfg["alarm_budget_per_100_series_months"]})
            if feasible is None and rate <= cfg["alarm_budget_per_100_series_months"]:
                feasible = (thr, a)   # самый чувствительный порог в бюджете
            print(f"  {method} thr={thr}: calibration alarm rate {rate:.2f}/100", flush=True)
        if feasible is None:   # ни один порог не уложился — берём самый строгий и пишем об этом
            thr = cfg["thresholds"][method][-1] if method != "BOCPD" else cfg["thresholds"][method][-1]
            feasible = (thr, a)
        chosen[method] = feasible[0]
        base_alarms[method] = feasible[1]
    pd.DataFrame(tuning).to_csv(out / "threshold_tuning.csv", index=False)
    (out / "chosen_thresholds.json").write_text(json.dumps(chosen, indent=1))
    print("пороги:", chosen, flush=True)

    # --- A. semi-synthetic injection -----------------------------------------------------------
    inj = cfg["injection"]
    rows, case_rows = [], []
    onset_all = inj["onset_months"]
    for seed in range(inj["seeds"]):
        rng = np.random.default_rng(cfg["seed"] + seed)
        n_ev = int(round(inj["share_of_series"] * panel.n_series))
        idx = rng.choice(panel.n_series, n_ev, replace=False)
        onset = rng.choice(onset_all, n_ev)
        mag = rng.choice(inj["magnitudes_sigma"], n_ev)
        sign = rng.choice([-1, 1], n_ev)
        r2 = r.copy()
        # масштаб, использованный бы при z на момент onset (только прошлое)
        for i, o, m, s in zip(idx, onset, mag, sign):
            scale = np.nanmedian(np.abs(r[i, 6:o] - np.nanmedian(r[i, 6:o]))) * 1.4826
            scale = max(scale, 0.01)
            scale = np.sqrt((len(r[i, 6:o]) * scale ** 2 + 6 * cat_ref[o, cats[i]] ** 2) / (len(r[i, 6:o]) + 6))
            r2[i, o:] += s * m * scale
        z2 = standardize(r2[idx], cats[idx], cat_ref)
        is_event = np.zeros(panel.n_series, dtype=bool); is_event[idx] = True
        for method in METHODS:
            a_ev = run_detector_matrix(z2, method, chosen[method], cooldown=cfg["cooldown_months"], hazard=cfg["bocpd_hazard"])
            tp = 0; delays = []; n_alarms_ev = 0
            for k, (i, o, m) in enumerate(zip(idx, onset, mag)):
                alarm_t = [t for t in np.where(a_ev[k])[0] if t >= test[0]]
                hit = match_alarms(o, alarm_t, cfg["match_max_delay_months"])
                n_alarms_ev += len(alarm_t)
                tp += hit is not None
                if hit is not None:
                    delays.append(hit - o)
                case_rows.append({"seed": seed, "method": method, "magnitude": m, "detected": hit is not None})
            a_null = base_alarms[method][~is_event][:, test[0]:test[1] + 1]
            n_null_alarms = int(a_null.sum())
            n_alarms = n_alarms_ev + n_null_alarms
            p, rec, f1 = prf(tp, n_alarms, n_ev)
            rows.append({"seed": seed, "method": method, "threshold": chosen[method], "n_events": n_ev,
                         "n_alarms": n_alarms, "event_precision": p, "event_recall": rec, "event_f1": f1,
                         "median_delay_months": float(np.median(delays)) if delays else np.nan,
                         "false_alarms_per_100_series_months": 100 * a_null.mean()})
        print(f"  injection seed {seed} {time.time() - t0:.0f}s", flush=True)
    inj_df = pd.DataFrame(rows)
    inj_df.to_csv(out / "injection_runs.csv", index=False)
    summary = inj_df.groupby("method").agg(["mean", "std"]).drop(columns=["seed", "n_events"], level=0)
    summary.to_csv(out / "injection_summary.csv")
    by_mag = pd.DataFrame(case_rows).groupby(["method", "magnitude"]).detected.mean().rename("recall").reset_index()
    by_mag.to_csv(out / "injection_recall_by_magnitude.csv", index=False)

    # --- B. реестр реальных событий ----------------------------------------------------------------
    registry = pd.read_csv(ROOT / "data/inputs/event_registry.csv", parse_dates=["event_date"])
    meta = panel.meta
    reg_rows = []
    for ev in registry.itertuples():
        t_event = int(panel.months.get_indexer([ev.event_date.to_period("M").to_timestamp()])[0])
        patterns = [x for x in ev.mo_name_patterns.split(";") if x]
        for pat in patterns:
            sel = meta[meta.mo_name.fillna("").str.contains(pat) &
                       meta.region_name.fillna("").apply(lambda x: any(rg in x for rg in ev.regions.split(";")))]
            ids = sorted(sel.territory_id.unique())
            reg_rows.append({"event_id": ev.event_id, "pattern": pat, "matched_territories": len(ids),
                             "territory_ids": ";".join(map(str, ids)), "t_event": t_event,
                             "evaluable": bool(cat_ref.shape[0] > t_event >= cal[0])})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "event_registry_matching.csv", index=False)
    detail = []
    for row in reg[(reg.matched_territories > 0) & reg.evaluable].itertuples():
        te = row.t_event
        ids = [int(x) for x in row.territory_ids.split(";")]
        for method in METHODS:
            a = base_alarms[method]
            window = slice(te, min(te + cfg["match_max_delay_months"], 23) + 1)
            # базовая частота: доля всех рядов с тревогой в том же окне
            base_rate = a[:, window].any(1).mean()
            for tid in ids:
                for sid in np.where((meta.territory_id == tid).to_numpy())[0]:
                    detail.append({"event_id": row.event_id, "pattern": row.pattern, "territory_id": tid,
                                   "category": meta.category[sid], "method": method,
                                   "alarm_in_window": bool(a[sid, window].any()),
                                   "max_abs_z_in_window": float(np.nanmax(np.abs(z[sid, window]))),
                                   "population_alarm_rate_same_window": float(base_rate)})
    detail_df = pd.DataFrame(detail)
    detail_df.to_csv(out / "event_registry_detection.csv", index=False)

    # --- C. offline-proxy разметка (только чувствительность) ----------------------------------
    prox = cfg["offline_proxy"]
    proxy_rows = []
    full = np.where(np.isfinite(z), z, 0.)
    valid_cols = np.where(np.isfinite(z).any(0))[0]
    zc = z[:, valid_cols]
    sub = np.random.default_rng(cfg["seed"]).choice(panel.n_series, 3000, replace=False)
    for mult in prox["penalty_multipliers"]:
        events = {}
        for i in sub:
            cuts = offline_mean_changes(zc[i], mult * np.log(zc.shape[1]), prox["min_segment"])
            keep = []
            for c in cuts:
                before, after = zc[i, :c], zc[i, c:]
                if abs(after.mean() - before.mean()) >= prox["min_shift_sigma"]:
                    keep.append(int(valid_cols[c]))
            events[i] = [c for c in keep if c >= test[0]]
        for method in METHODS:
            a = base_alarms[method]
            tp = n_alarm = n_event = 0
            delays = []
            for i, evs in events.items():
                al = [t for t in np.where(a[i])[0] if t >= test[0]]
                n_alarm += len(al); n_event += len(evs)
                used = set()
                for e in evs:
                    hit = match_alarms(e, [t for t in al if t not in used], cfg["match_max_delay_months"])
                    if hit is not None:
                        used.add(hit); tp += 1; delays.append(hit - e)
            p, rec, f1 = prf(tp, n_alarm, n_event)
            proxy_rows.append({"penalty_multiplier": mult, "method": method, "n_proxy_events": n_event,
                               "n_alarms": n_alarm, "precision": p, "recall": rec, "f1": f1,
                               "median_delay_months": float(np.median(delays)) if delays else np.nan,
                               "evaluation": "offline_heuristic_labels_not_verified_shocks"})
    pd.DataFrame(proxy_rows).to_csv(out / "offline_proxy_sensitivity.csv", index=False)

    # --- примеры для презентации -----------------------------------------------------------------
    np.save(out / "z.npy", z.astype(np.float32))
    for method in METHODS:
        np.save(out / f"alarms_{method}.npy", base_alarms[method])
    manifest = {"run": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - t0,
                "thresholds": chosen, "config": cfg,
                "note": "Метки событий — синтетические инъекции, реестр из 7 событий и offline-proxy; верифицированной разметки шоков по МО нет."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(summary.round(3).to_string())
    print("готово", time.time() - t0)


if __name__ == "__main__":
    main()
