"""Явно сезонный Prophet: dev-подбор на общей подвыборке, без изменения исходной базы."""
import argparse
import hashlib
import json
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sber-research-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.mun_data import load_municipal
from src.mun_eval import make_grid, score_model


def fit_job(args):
    variant, o, ids, values, months = args
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    future = pd.DataFrame({"ds": pd.date_range(months[o] + pd.DateOffset(months=1), periods=12, freq="MS")})
    out, failures = [], 0
    for sid, y in zip(ids, values):
        try:
            model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                            **{k: v for k, v in variant.items() if k != "id"})
            model.fit(pd.DataFrame({"ds": months[:o+1], "y": y[:o+1]}))
            pred = np.maximum(0., model.predict(future).yhat.to_numpy())
        except (ValueError, RuntimeError):
            pred = np.repeat(y[o], 12)
            failures += 1
        out.append((sid, pred))
    return variant["id"], o, out, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs/research_audit.json").read_text())
    panel, _ = load_municipal(ROOT)
    ids = np.sort(np.random.default_rng(cfg["seed"]).choice(panel.meta.territory_id.unique(),
                         cfg["prophet_sample_municipalities"], replace=False))
    rows = np.where(panel.meta.territory_id.isin(ids))[0]
    outdir = ROOT / "artifacts/research_prophet"
    outdir.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256((ROOT / "data/inputs/municipal_consumption.parquet").read_bytes()).hexdigest()
    meta_path = outdir / "manifest.json"
    specification = {"data_sha256": fingerprint, "sample_territories": ids.tolist(),
                     "variants": cfg["prophet_variants"], "status": cfg["evaluation_status"]}
    if meta_path.exists() and all((outdir / f'{v["id"]}.npy').exists() for v in cfg["prophet_variants"]):
        old = json.loads(meta_path.read_text())
        if all(old.get(k) == v for k, v in specification.items()):
            print("Prophet research: совместимый кэш найден", flush=True)
            return
    predictions = {v["id"]: np.full((24, 13, panel.n_series), np.nan, dtype=np.float32)
                   for v in cfg["prophet_variants"]}
    jobs = [(v, o, chunk, panel.values[chunk], panel.months)
            for v in cfg["prophet_variants"] for o in range(5, 24)
            for chunk in np.array_split(rows, max(1, len(rows)//25))]
    failures = 0
    with Pool(args.workers) as pool:
        for k, (name, o, result, failed) in enumerate(pool.imap_unordered(fit_job, jobs)):
            failures += failed
            for sid, pred in result:
                predictions[name][o, 1:, sid] = pred
            if k % 12 == 0:
                print(f"Prophet seasonal: {k+1}/{len(jobs)} chunks", flush=True)
    from src.mun_data import MunicipalPanel
    subset = MunicipalPanel(panel.values[rows], panel.meta.iloc[rows], panel.months)
    metrics = []
    for name, arr in predictions.items():
        np.save(outdir / f"{name}.npy", arr)
        for stage in ("validation", "test"):
            metrics.append(score_model(subset, arr[:, :, rows], make_grid(), stage, name))
    metrics = pd.concat(metrics)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    selected = metrics[metrics.stage == "validation"].groupby("model").mae.mean().idxmin()
    specification.update(selected_on_validation=selected, fallbacks=failures,
                         sample_series=len(rows), fits=len(rows)*19*len(predictions),
                         note="25 случайных МО; все сравнения на той же выборке; официальный baseline неизвестен")
    meta_path.write_text(json.dumps(specification, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
