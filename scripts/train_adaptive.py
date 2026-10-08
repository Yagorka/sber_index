"""Новые эксперименты. Старый holdout только recheck; новым независимым не считается."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecasting import load_panel, fit_predict, score_predictions
from src.adaptive import fit_adaptive_catboost, calibration_weights, mix_stream, correct_bias, reconcile
from src.transitions import (residual_stream, detect_stream, synthetic_scenarios, benchmark_detector,
                             offline_mean_changes, match_events)
from src.municipal import sha256
from scripts.train_forecasts import find_statsmodels_python, write_json


def main():
    started = time.monotonic()
    cfg_path = ROOT / "configs/adaptive_forecast.json"
    cfg = json.loads(cfg_path.read_text())
    base = ROOT / json.loads((ROOT / cfg["baseline_pointer"]).read_text())["directory"]
    base_manifest = json.loads((base / "manifest.json").read_text())
    base_cfg = json.loads((base / "config.json").read_text())
    base_selection = json.loads((base / "frozen_selection.json").read_text())
    panel = load_panel(ROOT / base_cfg["input"])
    if sha256(ROOT / base_cfg["input"]) != base_manifest["data_sha256"]:
        raise ValueError("Источник изменился: сначала обновите baseline")
    run_id = datetime.now(timezone.utc).strftime("adaptive_%Y%m%dT%H%M%S_%fZ")
    run = ROOT / cfg["output_root"] / run_id
    (run / "models").mkdir(parents=True)
    write_json(run / "config.json", cfg)
    panel.to_csv(run / "target_panel.csv", index_label="period")
    calibration = cfg["calibration_origins"]; validation = cfg["validation_origins"]
    horizons = base_cfg["validation"]["horizons_months"]
    assert calibration[1] + max(horizons) < validation[0]
    assert validation[1] + max(horizons) < len(panel) - 12
    cutoff = calibration[1] + max(horizons)
    origins = range(35, len(panel))
    stats_python, stats_version = find_statsmodels_python(base_cfg)
    if stats_python is None: raise RuntimeError("Нужен statsmodels")
    print(f"RUN {run_id}; calibration targets до {panel.index[cutoff].date()}; validation origins {validation}", flush=True)
    residuals = residual_stream(panel, cfg["detectors"])
    residuals.to_csv(run / "residuals.csv", index=False)
    synthetic_calib = synthetic_scenarios(cfg["detectors"]["synthetic_calibration_scenarios"], cfg["detectors"], cfg["seed"])
    synthetic_test = synthetic_scenarios(cfg["detectors"]["synthetic_test_scenarios"], cfg["detectors"], cfg["seed"] + 1)
    tuning, synthetic_scores, detector_selection, streams = [], [], {}, []
    for method, thresholds in cfg["detectors"]["thresholds"].items():
        candidates = []
        for threshold in thresholds:
            trace = detect_stream(residuals, method, threshold, cfg["detectors"])
            real_calib = trace[trace["index"] <= cutoff]
            alarm_rate = 100 * real_calib.alarm.mean()
            score, _ = benchmark_detector(method, threshold, synthetic_calib, cfg["detectors"])
            score.update(real_calibration_alarm_rate=alarm_rate,
                         within_alarm_budget=alarm_rate <= cfg["detectors"]["alarm_budget_per_100_observations"])
            candidates.append(score); tuning.append(score)
        feasible = [c for c in candidates if c["within_alarm_budget"]]
        selected = max(feasible or candidates, key=lambda c: (c["event_f1"] if feasible else -c["real_calibration_alarm_rate"]))
        detector_selection[method] = selected
        trace = detect_stream(residuals, method, selected["threshold"], cfg["detectors"])
        streams.append(trace)
        score, cases = benchmark_detector(method, selected["threshold"], synthetic_test, cfg["detectors"])
        score["evaluation"] = "synthetic_independent_seed_only"
        synthetic_scores.append(score)
        cases.to_csv(run / f"synthetic_cases_{method}.csv", index=False)
        print(f"{method}: threshold {selected['threshold']}, synthetic F1 {score['event_f1']:.3f}", flush=True)
    alarms = pd.concat(streams, ignore_index=True)
    alarms.to_csv(run / "detector_stream.csv", index=False)
    pd.DataFrame(tuning).to_csv(run / "detector_tuning.csv", index=False)
    pd.DataFrame(synthetic_scores).to_csv(run / "synthetic_detector_metrics.csv", index=False)
    # Независимая от детекторов ретроспективная proxy-разметка годового роста.
    refs, proxy_scores = [], []
    for series in panel.columns:
        growth = np.log(panel[series] / panel[series].shift(12)).dropna().to_numpy()
        variance = max(float(np.var(growth[:cutoff - 11])), 1e-6)
        for multiplier in cfg["detectors"]["reference_penalty_multipliers"]:
            cuts = offline_mean_changes(growth, multiplier * variance * np.log(len(growth)), cfg["detectors"]["reference_min_segment"])
            refs.extend({"series_id": series, "index": c + 12, "period": panel.index[c + 12], "penalty_multiplier": multiplier} for c in cuts)
    reference = pd.DataFrame(refs, columns=["series_id", "index", "period", "penalty_multiplier"])
    reference.to_csv(run / "offline_proxy_events.csv", index=False)
    for multiplier in cfg["detectors"]["reference_penalty_multipliers"]:
        for method in detector_selection:
            tp = ne = na = 0; delays = []
            for series in panel.columns:
                events = reference[(reference.series_id == series) & (reference.penalty_multiplier == multiplier) & (reference["index"] > cutoff)]["index"].tolist()
                notices = alarms[(alarms.series_id == series) & (alarms.method == method) & (alarms["index"] > cutoff) & alarms.alarm]["index"].tolist()
                pairs = match_events(events, notices, delay=3)
                tp += len(pairs); ne += len(events); na += len(notices); delays += [a - e for e, a in pairs]
            precision = tp / na if na else 0.; recall = tp / ne if ne else 0.
            proxy_scores.append({"method": method, "penalty_multiplier": multiplier, "n_proxy_events": ne,
                                 "n_alarms": na, "proxy_precision": precision, "proxy_recall": recall,
                                 "proxy_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                                 "median_delay_months": np.median(delays) if delays else None,
                                 "evaluation": "offline_heuristic_labels_not_verified_shocks"})
    pd.DataFrame(proxy_scores).to_csv(run / "real_proxy_sensitivity.csv", index=False)
    # Все новые базовые прогнозы — rolling, с метками не позже origin.
    rows, training_audit = [], []
    def append_output(output, origin, name):
        for h, series, value in output:
            target = origin + h
            rows.append({"model": name, "variant": name, "origin_index": origin, "origin": panel.index[origin],
                         "target_index": target, "target_date": panel.index[origin] + pd.DateOffset(months=h),
                         "horizon_months": h, "series_id": series, "y_pred": value,
                         "y_true": panel[series].iloc[target] if target < len(panel) else np.nan})
    for variant in cfg["catboost_variants"]:
        print(f"Обучение {variant['id']}", flush=True)
        for origin in origins:
            output, fitted, audit = fit_adaptive_catboost(panel, origin, horizons, variant, cfg["catboost_params"], cfg["seed"])
            append_output(output, origin, variant["id"]); training_audit += audit
            if origin == len(panel) - 1:
                joblib.dump({"fitted": fitted, "variant": variant, "series_order": list(panel.columns)}, run / "models" / f"{variant['id']}.joblib")
            if (origin - 34) % 15 == 0: print(f"  {origin - 34}/{len(origins)} origins", flush=True)
    for origin in origins:
        output, _, audit = fit_predict(panel, origin, horizons, {"id": "forest", "family": "RandomForestDirect", "params": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 3}}, cfg["seed"])
        append_output(output, origin, "forest"); training_audit += [{**a, "variant": "forest"} for a in audit]
        output, _, _ = fit_predict(panel, origin, horizons, {"id": "growth", "family": "SeasonalGrowth", "params": {}}, cfg["seed"])
        append_output(output, origin, "growth")
    jobs = []
    for window in cfg["ets_windows"]:
        name = "hw_full" if window is None else f"hw_w{window}"
        for origin in origins:
            for series in panel.columns:
                jobs.append({"variant": name, "params": {"seasonal": "add", "damped_trend": False},
                             "origin_index": origin, "series_id": series,
                             "values": panel[series].iloc[max(0, origin + 1 - window) if window else 0:origin + 1].tolist()})
    write_json(run / "ets_jobs.json", jobs)
    subprocess.run([stats_python, str(ROOT / "scripts/holtwinters_worker.py"), str(run / "ets_jobs.json"), str(run / "ets_results.json")], check=True)
    ets_results = json.loads((run / "ets_results.json").read_text())
    for record in ets_results:
        append_output([(h, record["series_id"], record["forecast"][h - 1]) for h in horizons], record["origin_index"], record["variant"])
    pd.DataFrame(training_audit).to_csv(run / "training_audit.csv", index=False)
    bases = pd.DataFrame(rows)
    bases.to_csv(run / "base_prediction_stream.csv", index=False)
    calibration_rows = bases[bases.origin_index.between(*calibration)].copy()
    calibration_rows["ae"] = (calibration_rows.y_pred - calibration_rows.y_true).abs()
    cal_rank = calibration_rows.groupby("model").ae.mean().sort_values()
    cb_id = cal_rank[cal_rank.index.str.startswith("cb_")].idxmin()
    hw_id = cal_rank[cal_rank.index.str.startswith("hw_")].idxmin()
    ids = [cb_id, hw_id, "forest", "growth"]
    weights = calibration_weights(bases, ids, calibration)
    static, static_log = mix_stream(bases, ids, "CalibratedStatic", fixed_weights=weights)
    # N01 неизменный reference; не участвует в новом выборе параметров.
    old_weights = {int(h): {"cb_full": w["catboost_d3"], "forest": w["forest"], "hw_full": w["hw_add"]} for h, w in base_selection["ensemble_weights"].items()}
    old, _ = mix_stream(bases, ["cb_full", "forest", "hw_full"], "BaselineN01", fixed_weights=old_weights)
    derived, history_audit = [static, old], [static_log]
    for window in cfg["adaptive_error_windows"]:
        stream, log = mix_stream(bases, ids, f"Adaptive{window}", window=window, min_errors=cfg["minimum_matured_errors"])
        derived.append(stream); history_audit.append(log)
        for bias_window in cfg["bias_windows"]:
            corrected, log = correct_bias(stream, f"Adaptive{window}_Bias{bias_window}", bias_window, cfg["bias_shrinkage"], cfg["minimum_matured_errors"])
            derived.append(corrected); history_audit.append(log)
    for window in cfg["bias_windows"]:
        corrected, log = correct_bias(static, f"Static_Bias{window}", window, cfg["bias_shrinkage"], cfg["minimum_matured_errors"])
        derived.append(corrected); history_audit.append(log)
    total_boc = alarms[(alarms.series_id == "Всего") & (alarms.method == "BOCPD")].set_index("index").alarm
    gate, log = mix_stream(bases, ids, "RegimeAdaptive", regime=total_boc, min_errors=cfg["minimum_matured_errors"])
    derived.append(gate); history_audit.append(log)
    # Рецепт для reconciliation выбран на calibration; cov доступна к validation.
    eligible = pd.concat(derived, ignore_index=True)
    cal = eligible[eligible.origin_index.between(*calibration) & (eligible.model != "BaselineN01")].copy()
    cal["ae"] = (cal.y_pred - cal.y_true).abs()
    reconcile_parent = cal.groupby("model").ae.mean().idxmin()
    parent = eligible[eligible.model == reconcile_parent]
    for method in ["bottom_up", "mint_shrink"]:
        coherent, covariance = reconcile(parent, f"{reconcile_parent}_{method}", calibration, method, cfg["reconciliation_covariance_shrinkage"])
        derived.append(coherent)
        joblib.dump(covariance, run / "models" / f"covariance_{method}.joblib")
    all_predictions = pd.concat([bases, *derived], ignore_index=True)
    # Никакие оценки validation/recheck не участвуют в предварительной настройке.
    stages = []
    for stage, mask in [
        ("calibration", all_predictions.origin_index.between(*calibration)),
        ("development_validation", all_predictions.origin_index.between(*validation)),
        ("recheck_seen_holdout", all_predictions.target_index.between(len(panel) - 12, len(panel) - 1)),
        ("future", all_predictions.origin_index == len(panel) - 1)]:
        part = all_predictions[mask].copy(); part["stage"] = stage; stages.append(part)
    staged = pd.concat(stages, ignore_index=True)
    metrics, per_series = score_predictions(staged[staged.stage != "future"])
    validation_rank = metrics[(metrics.stage == "development_validation") & (metrics.model != "BaselineN01")].groupby("model").macro_mae.mean().sort_values()
    winner = validation_rank.index[0]
    selection = {"selected_on_development_validation": winner, "expert_ids_selected_on_calibration": ids,
                 "static_weights": weights, "reconciliation_parent": reconcile_parent,
                 "calibration_maximum_target_index": cutoff, "validation_start_origin_index": validation[0],
                 "detector_selection": detector_selection, "frozen_at": datetime.now(timezone.utc).isoformat(),
                 "holdout_status": cfg["holdout_status"], "independent_real_future_test_available": False}
    write_json(run / "selection.json", selection)
    staged.to_csv(run / "predictions.csv", index=False)
    metrics.to_csv(run / "metrics.csv", index=False); per_series.to_csv(run / "per_series_metrics.csv", index=False)
    pd.concat(history_audit, ignore_index=True).to_csv(run / "adaptive_history_audit.csv", index=False)
    subprocess.run([stats_python, str(ROOT / "scripts/markov_states_worker.py"), str(run / "target_panel.csv"), str(run), "--cut", str(cutoff)], check=True)
    ranking = metrics.groupby(["stage", "model"]).macro_mae.mean().unstack("stage")
    ranking.to_csv(run / "ranking.csv")
    code_files = [ROOT / p for p in ["src/adaptive.py", "src/transitions.py", "scripts/train_adaptive.py", "scripts/markov_states_worker.py"]]
    manifest = {"run_id": run_id, "baseline_run": str(base.relative_to(ROOT)), "data_sha256": base_manifest["data_sha256"],
                "config_sha256": sha256(cfg_path), "code_sha256": hashlib.sha256(''.join(sha256(p) for p in code_files).encode()).hexdigest(),
                "selected_model": winner, "statsmodels_version": stats_version, "holdout_status": cfg["holdout_status"],
                "elapsed_seconds": time.monotonic() - started, "n_base_models": bases.model.nunique(),
                "n_total_models": all_predictions.model.nunique(), "markov_parameters_known_through": str(panel.index[cutoff].date()),
                "independent_future_test": False, "information_assumption": base_cfg['validation']['information_assumption']}
    write_json(run / "manifest.json", manifest)
    journal = metrics.copy(); journal["run_id"] = run_id; journal["baseline_run"] = base_manifest["run_id"]
    journal["test_status"] = cfg["holdout_status"]
    path = ROOT / "tracking/adaptive_forecast_metrics.csv"
    journal.to_csv(path, mode="a", header=not path.exists(), index=False)
    write_json(ROOT / cfg["output_root"] / "latest.json", {"run_id": run_id, "directory": str(run.relative_to(ROOT))})
    print(ranking.sort_values("development_validation").round(3).to_string(), flush=True)
    print(f"Выбрана на development: {winner}; готово {run.relative_to(ROOT)}", flush=True)


if __name__ == "__main__": main()
