"""Обучение кандидатов, отбор только на dev, rolling holdout и финальные модели."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecasting import (load_panel, make_split, fit_predict, score_predictions,
                             inverse_error_ensemble, block_bootstrap_comparison)
from src.municipal import sha256


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def find_statsmodels_python(config):
    candidates = [sys.executable]
    requested = config.get("statsmodels_python", "auto")
    if requested != "auto":
        candidates.insert(0, requested)
    # Основной сценарий: statsmodels установлен в той же среде (environment.yml);
    # иначе путь к интерпретатору задаётся в configs/national_forecast.json: statsmodels_python.
    for candidate in dict.fromkeys(candidates):
        if not Path(candidate).exists():
            continue
        result = subprocess.run([candidate, "-c", "import statsmodels; print(statsmodels.__version__)"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return candidate, result.stdout.strip()
    return None, None


def statsmodels_forecasts(panel, pairs, variants, run, stage, python):
    jobs = []
    for variant in variants:
        for origin in sorted(set(o for o, _ in pairs)):
            for series in panel.columns:
                job = {"variant": variant["id"], "params": variant["params"], "origin_index": origin,
                       "series_id": series, "values": panel[series].iloc[:origin + 1].tolist()}
                if stage == "future":
                    job["save_path"] = str(run / "models" / f"{variant['id']}_{panel.columns.get_loc(series)}.pickle")
                jobs.append(job)
    if not jobs:
        return []
    input_path, output_path = run / f"ets_jobs_{stage}.json", run / f"ets_results_{stage}.json"
    write_json(input_path, jobs)
    subprocess.run([python, str(ROOT / "scripts/holtwinters_worker.py"), str(input_path), str(output_path)], check=True)
    return json.loads(output_path.read_text())


def run_stage(panel, pairs, variants, config, run, stage, stats_python):
    rows, audits = [], []
    ets_variants = [v for v in variants if v["family"] == "HoltWinters"]
    ets_results = statsmodels_forecasts(panel, pairs, ets_variants, run, stage, stats_python) if ets_variants else []
    lookup = {(r["variant"], r["origin_index"], r["series_id"]): r for r in ets_results}
    origins = sorted(set(o for o, _ in pairs))
    for variant in variants:
        started = time.monotonic()
        print(f"{stage}: {variant['id']} — {len(origins)} origins", flush=True)
        for i, origin in enumerate(origins):
            horizons = sorted(h for o, h in pairs if o == origin)
            if variant["family"] == "HoltWinters":
                output = [(h, s, lookup[(variant["id"], origin, s)]["forecast"][h - 1])
                          for h in horizons for s in panel.columns]
                trained, audit = {}, []
            else:
                output, trained, audit = fit_predict(panel, origin, horizons, variant, config["seed"])
            if stage == "future" and trained:
                joblib.dump({"fitted": trained, "variant": variant, "series_order": list(panel.columns),
                             "origin": str(panel.index[origin]), "train_input": panel.to_numpy()},
                            run / "models" / f"{variant['id']}.joblib")
            for record in audit:
                record.update({"stage": stage, "variant": variant["id"]})
                audits.append(record)
            for h, series, value in output:
                if not np.isfinite(value):
                    raise ValueError(f"Неконечный прогноз: {variant['id']}")
                target = panel.index[origin] + pd.DateOffset(months=h)
                actual = float(panel[series].iloc[origin + h]) if origin + h < len(panel) else np.nan
                fallback = lookup[(variant["id"], origin, series)]["fallback"] if variant["family"] == "HoltWinters" else False
                rows.append({"stage": stage, "variant": variant["id"], "model": variant["family"],
                             "series_id": series, "origin": panel.index[origin], "target_date": target,
                             "horizon_months": h, "y_true": actual, "y_pred": value, "fallback": fallback})
            if (i + 1) % 10 == 0:
                print(f"  {variant['id']}: {i + 1}/{len(origins)}", flush=True)
        print(f"  Время: {time.monotonic() - started:.1f} с", flush=True)
    if audits:
        pd.DataFrame(audits).to_csv(run / f"training_audit_{stage}.csv", index=False)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/national_forecast.json")
    args = parser.parse_args()
    started = time.monotonic()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    panel = load_panel(ROOT / config["input"])
    split, dev_pairs, test_pairs, holdout_start = make_split(panel, config["validation"])
    run_id = datetime.now(timezone.utc).strftime("national_%Y%m%dT%H%M%S_%fZ")
    run = ROOT / config["output_root"] / run_id
    (run / "models").mkdir(parents=True)
    write_json(run / "config.json", config)
    split.to_csv(run / "split_manifest.csv", index=False)
    panel.to_csv(run / "target_panel.csv", index_label="period")
    stats_python, stats_version = find_statsmodels_python(config)
    variants = [v for v in config["variants"] if v["family"] != "HoltWinters" or stats_python]
    model_status = [{"model": family, "status": "available", "reason": ""}
                    for family in sorted(set(v["family"] for v in variants))]
    for optional in config["optional_models"]:
        model_status.append({"model": optional, "status": "not_run",
                             "reason": "Пакет/checkpoint недоступен в текущем окружении; результат не имитируется"})
    if not stats_python:
        model_status.append({"model": "HoltWinters", "status": "not_run", "reason": "statsmodels отсутствует"})
    pd.DataFrame(model_status).to_csv(run / "model_status.csv", index=False)
    code_files = [ROOT / "src/forecasting.py", ROOT / "src/evaluation.py", ROOT / "src/consumer.py",
                  ROOT / "scripts/train_forecasts.py", ROOT / "scripts/holtwinters_worker.py"]
    code_hash = hashlib.sha256("".join(sha256(p) for p in code_files).encode()).hexdigest()
    manifest = {"run_id": run_id, "input": config["input"], "data_sha256": sha256(ROOT / config["input"]),
                "config_sha256": sha256(config_path), "code_sha256": code_hash,
                "split_sha256": sha256(run / "split_manifest.csv"), "python": sys.version,
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]},
                "statsmodels_python": stats_python, "statsmodels_version": stats_version,
                "geography": "Россия", "axis": "расходные категории, не муниципалитеты",
                "first_month": str(panel.index[0].date()), "last_month": str(panel.index[-1].date()),
                "holdout_start": str(panel.index[holdout_start].date()), "development_origins": len(set(o for o, _ in dev_pairs)),
                "holdout_target_months": config["validation"]["holdout_months"], "n_series": len(panel.columns),
                "information_assumption": config["validation"]["information_assumption"],
                "prophet_compared": False, "fundamental_models_run": False}
    write_json(run / "manifest.json", manifest)
    print(f"RUN: {run_id}; 5 рядов × {len(panel)} месяцев", flush=True)
    print(f"Dev origins: {len(set(o for o, _ in dev_pairs))}; holdout с {panel.index[holdout_start].date()}", flush=True)
    development = run_stage(panel, dev_pairs, variants, config, run, "development", stats_python)
    dev_metrics, _ = score_predictions(development)
    ranking = dev_metrics.groupby(["model", "variant"]).macro_mae.mean().reset_index().sort_values("macro_mae")
    best_by_family = ranking.drop_duplicates("model")
    chosen_ids = best_by_family.variant.tolist()
    chosen_variants = [v for v in variants if v["id"] in chosen_ids]
    development = development[development.variant.isin(chosen_ids)].copy()
    top_ids = best_by_family.head(config["ensemble"]["top_families"]).variant.tolist()
    weights = {}
    for h in config["validation"]["horizons_months"]:
        errors = dev_metrics[(dev_metrics.horizon_months == h) & dev_metrics.variant.isin(top_ids)].set_index("variant").macro_mae
        inverse = 1 / errors.clip(lower=1e-10)
        weights[str(h)] = (inverse / inverse.sum()).to_dict()
    development = pd.concat([development, inverse_error_ensemble(development, top_ids, weights)], ignore_index=True)
    metrics_dev, _ = score_predictions(development)
    selected_family = metrics_dev.groupby("model").macro_mae.mean().idxmin()
    selection = {"frozen_before_holdout_at": datetime.now(timezone.utc).isoformat(),
                 "best_variant_per_family": best_by_family.to_dict("records"),
                 "ensemble_variants": top_ids, "ensemble_weights": weights,
                 "selected_model_on_development": selected_family,
                 "rule": config["validation"]["selection_metric"],
                 "holdout_hyperparameters_changed": False}
    write_json(run / "frozen_selection.json", selection)
    # Публикуем все dev-результаты; выбор завершён прежде первого обращения к holdout-ошибкам.
    dev_metrics.to_csv(run / "all_variant_development_metrics.csv", index=False)
    print(f"Зафиксирована модель: {selected_family}. Начинается независимый holdout.", flush=True)
    holdout = run_stage(panel, test_pairs, chosen_variants, config, run, "holdout", stats_python)
    holdout = pd.concat([holdout, inverse_error_ensemble(holdout, top_ids, weights)], ignore_index=True)
    predictions = pd.concat([development, holdout], ignore_index=True)
    predictions["run_id"] = run_id
    predictions.to_csv(run / "predictions.csv", index=False)
    metrics, per_series = score_predictions(predictions)
    metrics.to_csv(run / "metrics.csv", index=False)
    per_series.to_csv(run / "per_series_metrics.csv", index=False)
    ranking_final = metrics.groupby(["stage", "model"]).macro_mae.mean().reset_index().rename(columns={"macro_mae": "mean_macro_mae"})
    ranking_final.to_csv(run / "ranking.csv", index=False)
    comparisons = [block_bootstrap_comparison(holdout, selected_family, baseline,
                    config["validation"]["bootstrap_replicates"], config["validation"]["bootstrap_block_months"], config["seed"])
                   for baseline in ["SeasonalNaive", "SeasonalGrowth"]]
    pd.DataFrame(comparisons).to_csv(run / "paired_bootstrap.csv", index=False)
    future_pairs = [(len(panel) - 1, h) for h in config["validation"]["horizons_months"]]
    future = run_stage(panel, future_pairs, chosen_variants, config, run, "future", stats_python)
    future = pd.concat([future, inverse_error_ensemble(future, top_ids, weights)], ignore_index=True)
    future["run_id"] = run_id
    future.to_csv(run / "future_forecasts.csv", index=False)
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
    manifest["holdout_evaluated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["holdout_status"] = "evaluated_once_after_freeze"
    manifest["selected_model"] = selected_family
    write_json(run / "manifest.json", manifest)
    # Отдельный журнал национального опыта не смешивается с муниципальным leaderboard.
    journal = metrics.copy()
    for col, value in {"run_id": run_id, "data_hash": manifest["data_sha256"],
                       "config_hash": manifest["config_sha256"], "split_id": manifest["split_sha256"],
                       "geography": "Россия", "unit": "млрд руб.", "series_axis": "category"}.items():
        journal[col] = value
    journal_path = ROOT / "tracking/national_forecast_metrics.csv"
    journal.to_csv(journal_path, mode="a", header=not journal_path.exists(), index=False)
    write_json(ROOT / config["output_root"] / "latest.json", {"run_id": run_id, "directory": str(run.relative_to(ROOT))})
    print(ranking_final.to_string(index=False), flush=True)
    print(f"Готово: {run.relative_to(ROOT)}; время {manifest['elapsed_seconds']} с", flush=True)


if __name__ == "__main__":
    main()
