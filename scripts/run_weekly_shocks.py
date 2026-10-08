"""Детекторы сдвигов и раннее предупреждение на недельных реальных рядах СберИндекса.

Панель: 46 категорий роста расходов г/г (151 неделя, с 2023-11-12) и 5 возрастных групп
индекса потребительской активности (г/г, 299 недель, с 2021-01-10 → 247 недель г/г).
Метки событий — offline-сегментация по полным данным (proxy, НЕ верифицированные шоки):
она независима от онлайн-детекторов, но не истина. Метрики показываем с чувствительностью
к штрафу сегментации. Ключевая ставка ЦБ — внешний датированный признак (as-of).
Разделение по времени: пороги на калибровке, ранний сигнал — train → gap → test.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.consumer import read_export
from src.transitions import Detector, offline_mean_changes

METHODS = ["CUSUM", "PageHinkley", "BOCPD"]
CFG = {
    "seed": 42, "median_window": 12, "scale_window": 26, "min_scale_history": 8, "min_scale": 0.5,
    "warmup": 8, "cooldown": 4, "alarm_budget_per_100": 1.0, "match_max_delay": 6,
    "calibration_end_frac": 0.55, "test_start_frac": 0.70,
    "thresholds": {"CUSUM": [3, 4, 5, 6, 8, 10, 14, 20, 28, 40], "PageHinkley": [3, 5, 8, 12, 16, 24, 36, 50, 70], "BOCPD": [0.3, 0.5, 0.7, 0.9, 0.97]},
    "proxy": {"penalty_multipliers": [4, 8, 16, 32], "primary": 16, "min_segment": 8, "min_shift_sigma": 2.5},
    "early_warning": {"windows": [4, 13], "z_event_threshold": 2.0, "train_label_end_frac": 0.65, "gap_weeks": 8},
}


def load_series():
    src = json.loads((ROOT / "configs/consumer_sources.json").read_text())["datasets"]
    wk, _ = read_export(ROOT / src["weekly_growth"]["preferred_file"])
    wk["category"] = wk.category.str.strip()
    wk["period"] = pd.to_datetime(wk.period)
    weekly = wk.pivot(index="period", columns="category", values="value").sort_index()
    act, _ = read_export(ROOT / src["activity"]["preferred_file"])
    act["period"] = pd.to_datetime(act.period)
    act = act.assign(series=act.age.str.strip()).pivot(index="period", columns="series", values="value").sort_index()
    act_yoy = 100 * (act / act.shift(52) - 1)
    act_yoy.columns = ["Активность: " + c for c in act_yoy.columns]
    return weekly, act_yoy.dropna(how="all").dropna()


def online_z(x):
    """z_t по прошлому: медиана 12 нед. и робастный масштаб остатков за 26 нед."""
    n = len(x)
    z = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    w, sw = CFG["median_window"], CFG["scale_window"]
    for t in range(w, n):
        m = np.median(x[t - w:t])
        resid[t] = x[t] - m
        past = resid[max(w, t - sw):t]
        past = past[np.isfinite(past)]
        if len(past) >= CFG["min_scale_history"]:
            s = max(CFG["min_scale"], 1.4826 * np.median(np.abs(past - np.median(past))))
            z[t] = resid[t] / s
    return z


def run(z, method, thr):
    det = Detector(method, thr, CFG["cooldown"], 1 / 26)
    alarms = np.zeros(len(z), dtype=bool)
    scores = np.zeros(len(z))
    k = 0
    for t, v in enumerate(z):
        if not np.isfinite(v):
            continue
        sc, a = det.update(float(v), k)
        scores[t] = sc
        alarms[t] = a and k >= CFG["warmup"] // 2
        k += 1
    return alarms, scores


def proxy_events(x, mult, upto=None):
    x = np.asarray(x[:upto] if upto else x, dtype=float)
    sigma = max(1.4826 * np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / np.sqrt(2), 0.3)
    cuts = offline_mean_changes(x / sigma, mult * np.log(len(x)), CFG["proxy"]["min_segment"])
    keep = []
    for c in cuts:
        lo = max(0, c - CFG["proxy"]["min_segment"])
        before, after = x[lo:c] / sigma, x[c:c + CFG["proxy"]["min_segment"]] / sigma
        if abs(after.mean() - before.mean()) >= CFG["proxy"]["min_shift_sigma"]:
            keep.append(int(c))
    return keep


def match_one_to_one(events, alarms, delay):
    used, hits, delays = set(), 0, []
    for e in sorted(events):
        cand = [a for a in sorted(alarms) if a not in used and e <= a <= e + delay]
        if cand:
            used.add(cand[0]); hits += 1; delays.append(cand[0] - e)
    return hits, delays


def main():
    t0 = time.time()
    weekly, act = load_series()
    panels = {"weekly_growth": weekly, "activity_yoy": act}
    out = ROOT / "artifacts/weekly_shocks"
    out.mkdir(parents=True, exist_ok=True)
    rate = pd.read_csv(ROOT / "data/external/cbr_key_rate_daily.csv", parse_dates=["date"]).set_index("date").rate_pct
    results, tuning, proxy_rows, ew_rows, ew_curve, series_examples = [], [], [], [], [], {}
    all_z = {}
    for pname, df in panels.items():
        n = len(df)
        cal_end, test_start = int(CFG["calibration_end_frac"] * n), int(CFG["test_start_frac"] * n)
        Z = np.vstack([online_z(df[c].to_numpy(float)) for c in df.columns])
        all_z[pname] = (df, Z)
        chosen, alarms, scores = {}, {}, {}
        for method in METHODS:
            pick = None
            for thr in CFG["thresholds"][method]:
                res = [run(z, method, thr) for z in Z]
                A = np.vstack([r[0] for r in res])
                rate_cal = 100 * A[:, :cal_end][np.isfinite(Z[:, :cal_end])].mean()
                tuning.append({"panel": pname, "method": method, "threshold": thr, "alarm_rate_cal_per_100": rate_cal,
                               "alarm_rate_test_per_100": 100 * A[:, test_start:][np.isfinite(Z[:, test_start:])].mean()})
                if pick is None and rate_cal <= CFG["alarm_budget_per_100"]:
                    pick = (thr, A, np.vstack([r[1] for r in res]))
            if pick is None:
                thr = CFG["thresholds"][method][-1]
                pick = (thr, A, np.vstack([r[1] for r in res]))
            chosen[method], alarms[method], scores[method] = pick
        pd.DataFrame([{"panel": pname, "method": m, "threshold": chosen[m]} for m in METHODS]).to_csv(
            out / f"chosen_thresholds_{pname}.csv", index=False)
        # --- proxy-разметка и оценка на test ---
        for mult in CFG["proxy"]["penalty_multipliers"]:
            events = {c: [e for e in proxy_events(df[c].to_numpy(float), mult) if e >= test_start] for c in df.columns}
            for method in METHODS:
                tp = na = ne = 0; delays = []
                for i, c in enumerate(df.columns):
                    al = [t for t in np.where(alarms[method][i])[0] if t >= test_start]
                    h, d = match_one_to_one(events[c], al, CFG["match_max_delay"])
                    tp += h; na += len(al); ne += len(events[c]); delays += d
                p = tp / na if na else 0.; r = tp / ne if ne else 0.
                fa = 100 * (na - tp) / np.isfinite(Z[:, test_start:]).sum()
                proxy_rows.append({"panel": pname, "penalty_multiplier": mult, "method": method, "threshold": chosen[method],
                                   "n_proxy_events_test": ne, "n_alarms_test": na, "event_precision": p, "event_recall": r,
                                   "event_f1": 2 * p * r / (p + r) if p + r else 0.,
                                   "false_alarms_per_100_series_weeks": fa,
                                   "median_delay_weeks": float(np.median(delays)) if delays else np.nan,
                                   "labels": "offline_proxy_not_verified_shocks"})
        # --- примеры рядов для презентации ---
        series_examples[pname] = {"index": [str(x.date()) for x in df.index], "columns": list(df.columns)}
        np.save(out / f"z_{pname}.npy", Z.astype(np.float32))
        for m in METHODS:
            np.save(out / f"alarms_{m}_{pname}.npy", alarms[m])
            np.save(out / f"scores_{m}_{pname}.npy", scores[m].astype(np.float32))
        # --- раннее предупреждение ---
        ew = CFG["early_warning"]
        breadth = np.nanmean(np.abs(Z) > ew["z_event_threshold"], axis=0)
        rate_w = rate.reindex(df.index.union(rate.index)).ffill().reindex(df.index)
        rate_d13 = rate_w.diff(13).to_numpy()
        changes = rate.diff().fillna(0) != 0
        last_change = pd.Series(np.where(changes, rate.index, pd.NaT), index=rate.index).ffill()
        weeks_since = np.array([(d - last_change.reindex([d], method="ffill").iloc[0]).days / 7
                                if pd.notna(last_change.reindex([d], method="ffill").iloc[0]) else 99. for d in df.index])
        cut_train = int(ew["train_label_end_frac"] * n)
        for k in ew["windows"]:
            X_rows, y_rows, t_rows, s_rows = [], [], [], []
            for i, c in enumerate(df.columns):
                x = df[c].to_numpy(float)
                ev_train = proxy_events(x, CFG["proxy"]["primary"], upto=cut_train)
                ev_full = proxy_events(x, CFG["proxy"]["primary"])
                for t in range(CFG["median_window"] + CFG["min_scale_history"] + 2, n - k):
                    z_ = Z[i]
                    if not np.isfinite(z_[t]):
                        continue
                    zs = z_[max(0, t - 3):t + 1]
                    feats = [z_[t], z_[t - 1] if np.isfinite(z_[t - 1]) else 0., float(np.nanmean(zs)), float(np.nanmax(np.abs(zs))),
                             breadth[t], breadth[t] - breadth[t - 4], scores["CUSUM"][i, t], scores["PageHinkley"][i, t],
                             scores["BOCPD"][i, t], rate_d13[t] if np.isfinite(rate_d13[t]) else 0., weeks_since[t],
                             float(df.index[t].isocalendar().week)]
                    in_train = t + k <= cut_train - CFG["proxy"]["min_segment"]
                    events = ev_train if in_train else ev_full
                    label = int(any(t < e <= t + k for e in events))
                    X_rows.append(feats); y_rows.append(label); t_rows.append(t); s_rows.append(i)
            X, y, tt, ss = map(np.asarray, (X_rows, y_rows, t_rows, s_rows))
            train_m = tt + k <= cut_train - CFG["proxy"]["min_segment"]
            test_m = tt >= cut_train + ew["gap_weeks"]
            val_cut = np.quantile(tt[train_m], 0.8)
            fit_m = train_m & (tt <= val_cut)
            val_m = train_m & (tt > val_cut)
            if y[fit_m].sum() < 5 or y[test_m].sum() < 5:
                ew_rows.append({"panel": pname, "k_weeks": k, "status": "skipped_too_few_positive_labels",
                                "n_pos_fit": int(y[fit_m].sum()), "n_pos_test": int(y[test_m].sum())})
                continue
            models = {"HGB": HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.05,
                                                             l2_regularization=2.0, random_state=CFG["seed"]),
                      "Logistic": LogisticRegression(max_iter=500, C=0.5),
                      "BOCPD_score_only": None}
            prevalence = y[fit_m].mean()
            for name, model in models.items():
                if name == "BOCPD_score_only":
                    p_val, p_test = X[val_m][:, 8], X[test_m][:, 8]
                else:
                    scaler_mu, scaler_sd = X[fit_m].mean(0), X[fit_m].std(0) + 1e-9
                    xf = (X - scaler_mu) / scaler_sd if name == "Logistic" else X
                    model.fit(xf[fit_m], y[fit_m])
                    p_val = model.predict_proba(xf[val_m])[:, 1]
                    p_test = model.predict_proba(xf[test_m])[:, 1]
                # порог: максимальная чувствительность при частоте тревог <= бюджет на val
                budget = CFG["alarm_budget_per_100"] / 100
                thr = np.quantile(p_val, 1 - budget) if len(p_val) else 1.
                alarm = p_test >= thr
                yt = y[test_m]
                tp = int((alarm & (yt == 1)).sum())
                brier = brier_score_loss(yt, np.clip(p_test, 0, 1)) if name != "BOCPD_score_only" else np.nan
                ew_rows.append({"panel": pname, "k_weeks": k, "model": name, "status": "ok",
                                "n_test_rows": int(test_m.sum()), "test_prevalence": float(yt.mean()),
                                "pr_auc": float(average_precision_score(yt, p_test)),
                                "pr_auc_baseline_prevalence": float(yt.mean()),
                                "brier": brier, "brier_baseline_prevalence": float(brier_score_loss(yt, np.full(len(yt), prevalence))),
                                "alarm_threshold": float(thr), "alarm_rate_per_100": float(100 * alarm.mean()),
                                "row_precision": float(tp / alarm.sum()) if alarm.sum() else 0.,
                                "row_recall": float(tp / yt.sum()) if yt.sum() else 0.})
            print(f"  EW {pname} k={k}: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(tuning).to_csv(out / "threshold_tuning.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(out / "detector_metrics_proxy_labels.csv", index=False)
    pd.DataFrame(ew_rows).to_csv(out / "early_warning_metrics.csv", index=False)
    (out / "series_index.json").write_text(json.dumps(series_examples, ensure_ascii=False))
    (out / "config.json").write_text(json.dumps(CFG, ensure_ascii=False, indent=1))
    print(pd.DataFrame(proxy_rows).query(f"penalty_multiplier == {CFG['proxy']['primary']}").round(3).to_string())
    print(pd.DataFrame(ew_rows).round(3).to_string())
    print("готово", time.time() - t0)


if __name__ == "__main__":
    main()
